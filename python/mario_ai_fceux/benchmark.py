"""Create a comparable Markdown table from standardized policy evaluation JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
