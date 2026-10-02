#!/usr/bin/env python3
"""Champion gate: decide promote / archive / inconclusive for a candidate pin.

Every candidate checkpoint pin must match-or-beat the incumbent's 1-1 sweep on
the fixed seed suite before publish. Decision uses Wilson score intervals:

- promote     candidate lower bound strictly beats incumbent upper bound
- archive     candidate is strictly worse (candidate upper bound below
              incumbent lower bound, or candidate clears 0 where incumbent
              has clears at a decisive margin)
- inconclusive otherwise: gather more episodes; the incumbent stays champion

Evidence is one or more standardized results.json files per side (see
mario_ai_fceux.benchmark), matched on evaluation conditions. Cross-lane
comparison requires matched load; otherwise keep suites separate and never
pool across lanes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_benchmark_module():
    """Load benchmark.py by path: its package __init__ imports torch, which the
    gate does not need."""
    path = PROJECT_ROOT / "python" / "mario_ai_fceux" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("mario_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REQUIRED_CONDITIONS = _load_benchmark_module().REQUIRED_CONDITIONS

Z_95 = 1.959963984540054

CHAMPION_PIN = "9d4cd28215c806960d1b0c574dad229b8e058ec5de29fed3abe2b3b633153585"
CHAMPION_RECORD = {"victories": 15, "episodes": 77,
                   "note": "pooled evidence for 9d4cd282 across four sweeps"}


def wilson_interval(victories: int, episodes: int, z: float = Z_95) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not 0 <= victories <= episodes:
        raise ValueError("victories must be between 0 and episodes")
    proportion = victories / episodes
    denominator = 1.0 + z * z / episodes
    centre = (proportion + z * z / (2 * episodes)) / denominator
    margin = (z / denominator) * math.sqrt(
        proportion * (1 - proportion) / episodes + z * z / (4 * episodes * episodes))
    return max(0.0, centre - margin), min(1.0, centre + margin)


def summarize(reports: list[dict]) -> dict:
    """Pool sweep reports into victories/episodes with per-suite and pooled CIs."""
    suites = []
    victories = 0
    episodes = 0
    for report in reports:
        wins = int(report.get("victories", 0))
        played = int(report.get("episodes_finished", 0))
        victories += wins
        episodes += played
        suites.append({
            "world": report.get("world"),
            "level": report.get("level"),
            "seed": report.get("evaluation_seed"),
            "checkpoint_sha256": report.get("checkpoint_sha256"),
            "victories": wins,
            "episodes_finished": played,
            "wilson_95": wilson_interval(wins, played) if played else None,
        })
    return {
        "suites": suites,
        "victories": victories,
        "episodes": episodes,
        "wilson_95": wilson_interval(victories, episodes) if episodes else None,
    }


def check_comparable(candidate_reports: list[dict], incumbent_reports: list[dict]) -> list[str]:
    """Conditions must match across sides and within each side for pooling."""
    problems: list[str] = []
    all_reports = [("candidate", r) for r in candidate_reports] + \
                  [("incumbent", r) for r in incumbent_reports]
    for index, (side, report) in enumerate(all_reports):
        name = f"{side}-{index}"
        if report.get("evaluation_mode") != "greedy_no_learning":
            problems.append(f"{name}: evaluation must disable learning and exploration")
        episodes = report.get("episodes")
        finished = report.get("episodes_finished")
        victories = report.get("victories")
        if not isinstance(episodes, list) or not isinstance(finished, int) or finished < 0 \
                or len(episodes) != finished:
            problems.append(f"{name}: episode records do not match episodes_finished")
        elif sum(isinstance(item, dict) and item.get("reason") == "victory" for item in episodes) \
                != victories:
            problems.append(f"{name}: victory count does not match its episode records")
        if not report.get("checkpoint_sha256") and not report.get("weights_sha256"):
            problems.append(f"{name}: unattributed sweep (no checkpoint_sha256 or "
                            "weights_sha256); numbers cannot be credited to a pin")
    evaluators = {report.get("evaluator_sha256") for _, report in all_reports}
    if len(evaluators) > 1:
        problems.append("sweeps ran under different evaluator code (evaluator_sha256 mismatch)")
    reference = tuple(candidate_reports[0].get(key) for key in REQUIRED_CONDITIONS) \
        if candidate_reports else None
    for side, report in all_reports:
        if reference is not None and tuple(report.get(key) for key in REQUIRED_CONDITIONS) != reference:
            problems.append(f"{side}: evaluation conditions differ from candidate reference")
    shas = {report.get("checkpoint_sha256") for _, report in all_reports}
    if len(shas) > 2:
        problems.append("more than two distinct checkpoint_sha256 values in the comparison")
    return problems


def decide(candidate: dict, incumbent: dict) -> dict:
    """Apply the champion gate: promote, archive, or hold at inconclusive."""
    if not candidate.get("episodes"):
        return {"decision": "inconclusive", "reason": "candidate has no finished episodes"}
    if not incumbent.get("episodes"):
        return {"decision": "inconclusive",
                "reason": "incumbent has no finished episodes; establish the baseline first"}
    candidate_low, candidate_high = candidate["wilson_95"]
    incumbent_low, incumbent_high = incumbent["wilson_95"]
    if candidate_low > incumbent_high:
        return {"decision": "promote",
                "reason": "candidate lower bound beats incumbent upper bound"}
    if candidate_high < incumbent_low:
        return {"decision": "archive",
                "reason": "candidate is strictly worse than the incumbent"}
    return {"decision": "inconclusive",
            "reason": "confidence intervals overlap; run more episodes before promoting"}


def load_reports(paths: list[Path]) -> list[dict]:
    reports = []
    for path in paths:
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True,
                        help="results.json sweeps of the candidate pin")
    parser.add_argument("--incumbent", type=Path, nargs="*",
                        help="results.json sweeps of the incumbent pin")
    parser.add_argument("--use-recorded-champion", action="store_true",
                        help="use the recorded 9d4cd282 baseline (15/77) as the incumbent")
    parser.add_argument("--json", action="store_true", help="print the verdict as JSON")
    options = parser.parse_args()

    candidate_reports = load_reports(options.candidate)
    if options.use_recorded_champion:
        incumbent_summary = dict(CHAMPION_RECORD)
        incumbent_summary["wilson_95"] = wilson_interval(
            CHAMPION_RECORD["victories"], CHAMPION_RECORD["episodes"])
        incumbent_summary["suites"] = []
        incumbent_reports = []
    else:
        incumbent_reports = load_reports(options.incumbent or [])
        incumbent_summary = summarize(incumbent_reports)

    problems = check_comparable(candidate_reports, incumbent_reports)
    candidate_summary = summarize(candidate_reports)
    verdict = decide(candidate_summary, incumbent_summary)
    result = {
        "champion_pin": CHAMPION_PIN,
        "candidate": candidate_summary,
        "incumbent": incumbent_summary,
        "decision": verdict["decision"],
        "reason": verdict["reason"],
        "condition_problems": problems,
    }
    if problems:
        result["decision"] = "inconclusive"
        result["reason"] = "evaluation conditions are not comparable: " + "; ".join(problems)
    if options.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"decision: {result['decision']} ({result['reason']})")
        for side in ("candidate", "incumbent"):
            summary = result[side]
            interval = summary.get("wilson_95")
            rendered = f"[{interval[0]:.3f}, {interval[1]:.3f}]" if interval else "n/a"
            print(f"{side}: {summary['victories']}/{summary['episodes']} wilson95 {rendered}")
    return 0 if result["decision"] != "archive" else 2


if __name__ == "__main__":
    sys.exit(main())
