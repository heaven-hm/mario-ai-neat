"""Create a comparable Markdown table from standardized policy evaluation JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REQUIRED_CONDITIONS = ("rom_sha256", "fceux_sha256", "world", "level", "action_repeat_frames",
                       "start_protocol", "episodes_requested", "evaluation_mode",
                       "evaluation_seed")


def _finite_nonnegative_number(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except (OverflowError, TypeError):
        return False


def validate_reports(rows: list[tuple[str, dict[str, object]]]) -> None:
    """Reject comparisons whose evaluation conditions or completion differ."""
    if len(rows) < 2:
        raise ValueError("at least two policies are required for a comparison")
    names = [name for name, _ in rows]
    if len(names) != len(set(names)):
        raise ValueError("policy names must be unique")
    invalid_reports = [name for name, report in rows if not isinstance(report, dict)]
    if invalid_reports:
        raise ValueError("reports must be JSON objects: " + ", ".join(invalid_reports))
    reference = tuple(rows[0][1].get(key) for key in REQUIRED_CONDITIONS)
    mismatches: list[str] = []
    for name, report in rows:
        conditions = tuple(report.get(key) for key in REQUIRED_CONDITIONS)
        if any(value is None for value in conditions):
            mismatches.append(f"{name}: missing required ROM/emulator/start/evaluation metadata")
        elif conditions != reference:
            mismatches.append(f"{name}: benchmark conditions differ")
        requested = report.get("episodes_requested")
        finished = report.get("episodes_finished")
        if (not isinstance(requested, int) or isinstance(requested, bool) or requested <= 0
                or not isinstance(finished, int) or isinstance(finished, bool) or finished < 0):
            mismatches.append(f"{name}: requested/finished episode counts must be valid integers")
        elif finished != requested:
            mismatches.append(f"{name}: only {report.get('episodes_finished')} of "
                              f"{report.get('episodes_requested')} episodes finished")
        if report.get("evaluation_mode") != "greedy_no_learning":
            mismatches.append(f"{name}: evaluation must disable learning and exploration")
        algorithm = report.get("algorithm")
        if not isinstance(algorithm, str) or not algorithm.strip():
            mismatches.append(f"{name}: missing policy algorithm label")
        episodes = report.get("episodes")
        if not isinstance(episodes, list) or len(episodes) != finished:
            mismatches.append(f"{name}: episode records do not match the finished count")
        elif isinstance(episodes, list):
            malformed = []
            for index, item in enumerate(episodes, 1):
                if not isinstance(item, dict):
                    malformed.append(index)
                    continue
                max_x = item.get("max_x")
                decisions = item.get("action_decisions")
                elapsed = item.get("elapsed_seconds")
                if (item.get("reason") not in {"victory", "death", "timeout"}
                        or not _finite_nonnegative_number(max_x)
                        or not isinstance(decisions, int) or isinstance(decisions, bool) or decisions < 0
                        or not _finite_nonnegative_number(elapsed)):
                    malformed.append(index)
            if malformed:
                mismatches.append(f"{name}: malformed episode records at indices {malformed}")
            victories = sum(isinstance(item, dict) and item.get("reason") == "victory"
                            for item in episodes)
            reported_victories = report.get("victories")
            if (not isinstance(reported_victories, int) or isinstance(reported_victories, bool)
                    or victories != reported_victories):
                mismatches.append(f"{name}: victory count does not match its episode records")
            try:
                completion_rate = float(report.get("completion_rate", -1.0))
            except (TypeError, ValueError):
                completion_rate = -1.0
            expected_rate = victories / len(episodes) if episodes else 0.0
            if not math.isfinite(completion_rate) or abs(completion_rate - expected_rate) > 1e-9:
                mismatches.append(f"{name}: completion rate does not match its episode records")
    if mismatches:
        raise ValueError("incomparable evaluation reports:\n" + "\n".join(mismatches))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare policies from standardized SMB1 evaluations.")
    parser.add_argument("--result", action="append", required=True, metavar="NAME=PATH",
                        help="One standardized evaluation JSON report per policy.")
    parser.add_argument("--output", type=Path, default=Path("benchmark.md"))
    arguments = parser.parse_args()
    rows: list[tuple[str, dict[str, object]]] = []
    for value in arguments.result:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError("each --result must be NAME=PATH")
        report = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        rows.append((name, report))
    validate_reports(rows)
    lines = ["# SMB1 policy benchmark", "", "All policies must use the same ROM hash, FCEUX version, world, "
             "start state, episode count, action repeat, and evaluation-only mode.", "",
             "| Policy | Episodes | Wins | Completion | Mean X | Best X | Mean decisions | Mean seconds |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, report in rows:
        episodes = report.get("episodes", [])
        best_x = max((int(item.get("max_x", 0)) for item in episodes), default=0)
        count = max(1, len(episodes))
        mean_x = sum(float(item["max_x"]) for item in episodes) / count
        mean_decisions = sum(int(item["action_decisions"]) for item in episodes) / count
        mean_seconds = sum(float(item["elapsed_seconds"]) for item in episodes) / count
        lines.append(f"| {name} ({report['algorithm']}) | {len(episodes)} | {report['victories']} | "
                     f"{float(report['completion_rate']):.2%} | {mean_x:.1f} | {best_x} | "
                     f"{mean_decisions:.1f} | {mean_seconds:.1f} |")
    reference = rows[0][1]
    lines.extend(("", f"ROM SHA-256: `{reference['rom_sha256']}`  ",
                  f"FCEUX SHA-256: `{reference['fceux_sha256']}`  ",
                  f"World: {reference['world']} | Episodes: {reference['episodes_requested']} | "
                  f"Action repeat: {reference['action_repeat_frames']} | "
                  f"Evaluation seed: {reference['evaluation_seed']}"))
    arguments.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(arguments.output)


if __name__ == "__main__":
    main()
