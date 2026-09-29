"""Create a comparable Markdown table from standardized policy evaluation JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_CONDITIONS = ("rom_sha256", "fceux_sha256", "world", "action_repeat_frames",
                       "start_protocol", "episodes_requested", "evaluation_mode",
                       "evaluation_seed")


def validate_reports(rows: list[tuple[str, dict[str, object]]]) -> None:
    """Reject comparisons whose evaluation conditions or completion differ."""
    if len(rows) < 2:
        raise ValueError("at least two policies are required for a comparison")
    names = [name for name, _ in rows]
    if len(names) != len(set(names)):
        raise ValueError("policy names must be unique")
    reference = tuple(rows[0][1].get(key) for key in REQUIRED_CONDITIONS)
    mismatches: list[str] = []
    for name, report in rows:
        conditions = tuple(report.get(key) for key in REQUIRED_CONDITIONS)
        if any(value is None for value in conditions):
            mismatches.append(f"{name}: missing required ROM/emulator/start/evaluation metadata")
        elif conditions != reference:
            mismatches.append(f"{name}: benchmark conditions differ")
        if report.get("episodes_finished") != report.get("episodes_requested"):
            mismatches.append(f"{name}: only {report.get('episodes_finished')} of "
                              f"{report.get('episodes_requested')} episodes finished")
        if report.get("evaluation_mode") != "greedy_no_learning":
            mismatches.append(f"{name}: evaluation must disable learning and exploration")
        episodes = report.get("episodes")
        if not isinstance(episodes, list) or len(episodes) != report.get("episodes_finished"):
            mismatches.append(f"{name}: episode records do not match the finished count")
        elif isinstance(episodes, list):
            victories = sum(isinstance(item, dict) and item.get("reason") == "victory"
                            for item in episodes)
            if victories != report.get("victories"):
                mismatches.append(f"{name}: victory count does not match its episode records")
            completion_rate = float(report.get("completion_rate", -1.0))
            expected_rate = victories / len(episodes) if episodes else 0.0
            if abs(completion_rate - expected_rate) > 1e-9:
                mismatches.append(f"{name}: completion rate does not match its episode records")
    if mismatches:
        raise ValueError("incomparable evaluation reports:\n" + "\n".join(mismatches))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare NEAT, DDQN, PPO, and Rainbow evaluation reports.")
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
             "| Policy | Finished episodes | Victories | Completion rate | Best X |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, report in rows:
        episodes = report.get("episodes", [])
        best_x = max((int(item.get("max_x", 0)) for item in episodes), default=0)
        lines.append(f"| {name} | {report.get('episodes_finished', 0)} | {report.get('victories', 0)} | "
                     f"{float(report.get('completion_rate', 0.0)):.2%} | {best_x} |")
    arguments.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(arguments.output)


if __name__ == "__main__":
    main()
