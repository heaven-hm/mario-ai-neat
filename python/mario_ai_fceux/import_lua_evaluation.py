"""Convert clean-start Lua NEAT Champion logs into the shared benchmark JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from .environment import START_PROTOCOL

START_RE = re.compile(r"episode start \| .*?world=(\d+) \| level=(\d+) \| x=(\d+)")
END_RE = re.compile(r"episode end \| reason=([^|]+) \| .*?max_x=(\d+) \| frames=(\d+) \| decisions=(\d+) \| elapsed_seconds=(\d+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def champion_episodes(log_text: str, world: int, level: int) -> list[dict[str, object]]:
    """Read only complete episodes from the most recent Champion process run."""
    lines = log_text.splitlines()
    starts = [index for index, line in enumerate(lines)
              if " started | " in line and " | mode=champion |" in line]
    if not starts:
        raise ValueError("log has no Champion session; set PLAY_CHAMPION_ONLY=true and reload the Lua script")
    session = lines[starts[-1] + 1:]
    episodes: list[dict[str, object]] = []
    pending: tuple[int, int, int] | None = None
    for line in session:
        start_match = START_RE.search(line)
        if start_match:
            pending = tuple(int(value) for value in start_match.groups())
            continue
        end_match = END_RE.search(line)
        if not end_match or pending is None:
            continue
        episode_world, episode_level, _start_x = pending
        pending = None
        if (episode_world, episode_level) != (world, level):
            continue
        raw_reason = end_match.group(1).strip()
        reason = "timeout" if raw_reason == "stuck" else raw_reason
        if reason not in {"death", "victory", "timeout"}:
            continue
        max_x, _frames, decisions, elapsed = (int(value) for value in end_match.groups()[1:])
        episodes.append({"episode": len(episodes) + 1, "reason": reason,
                         "max_x": max_x, "terminal_x": max_x,
                         "action_decisions": decisions, "elapsed_seconds": float(elapsed)})
    return episodes


def build_report(log_path: Path, database_path: Path, rom_path: Path,
                 fceux_path: Path, world: int, level: int, requested: int,
                 evaluation_seed: int) -> dict[str, object]:
    if not 1 <= world <= 8 or not 1 <= level <= 4 or requested < 1:
        raise ValueError("world and level must be SMB1 ranges; episode count must be positive")
    episodes = champion_episodes(log_path.read_text(encoding="utf-8", errors="replace"), world, level)
    episodes = episodes[-requested:]
    if len(episodes) != requested:
        raise ValueError(f"found {len(episodes)} complete World {world}-{level} Champion episodes; "
                         f"need {requested}. Reload the updated script and run more clean-start episodes.")
    victories = sum(item["reason"] == "victory" for item in episodes)
    return {
        "algorithm": "Lua NEAT Champion (greedy, fixed 12-frame action repeat)",
        "checkpoint": str(database_path), "checkpoint_sha256": sha256_file(database_path),
        "source_revision": None, "world": world, "level": level,
        "evaluation_mode": "greedy_no_learning", "evaluation_seed": evaluation_seed,
        "rom_sha256": sha256_file(rom_path), "fceux_executable": str(fceux_path.resolve()),
        "fceux_sha256": sha256_file(fceux_path), "action_repeat_frames": 12,
        "start_protocol": START_PROTOCOL, "episodes_requested": requested,
        "episodes_finished": len(episodes), "victories": victories,
        "completion_rate": victories / len(episodes), "episodes": episodes,
        "lua_log": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--fceux", default="fceux")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--evaluation-seed", type=int, default=2026,
                        help="Recorded benchmark seed; Champion itself is deterministic.")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    fceux_path = Path(shutil.which(options.fceux) or options.fceux)
    report = build_report(options.log, options.database, options.rom, fceux_path,
                          options.world, options.level, options.episodes, options.evaluation_seed)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(options.output)


if __name__ == "__main__":
    main()
