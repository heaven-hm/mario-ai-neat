#!/usr/bin/env python3
"""Drift study: turn "continued training degrades clean performance" into a curve.

Method (agreed with CUSE-1, 2026-10-03):
  1. Start from an exact pin (model.pt + replay.npz pair, sha recorded).
  2. Train in short windows (--window-steps per window). The trainer's --steps
     bound is per-launch (transitions_received resets at each launch), so a
     relaunch per window exits gracefully at its own boundary.
  3. Pin model.pt + replay.npz at every window boundary (sha256 each).
  4. After ALL windows: one quiet-machine eval pass over every pin (t0 included),
     20 episodes each, deterministic, deployed selection path, CPU context recorded.
  5. Write curve.json: {step, wins, episodes, max_x stats, stack_mix, config_delta}
     per point. Judge every post-change window against the PIN's own measurement.

Always relaunch with --checkpoint-every 1000 so machine kills cost at most one
window (CUSE-1's lesson: their VM restarted 7+ times and only dense
checkpointing kept windows recoverable).

Labels per point: stack mix and config delta, per the project protocol. Eval
requirements per the project lesson: stop training actors first, verify 0
fceux+trainer CPU at eval time, record the CPU context in the record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def read_step(model: Path) -> int | None:
    try:
        import torch
        d = torch.load(model, map_location="cpu", weights_only=False)
        return int(d.get("steps"))
    except Exception:
        return None


def pin_pair(run_dir: Path, pin_dir: Path, label: str) -> dict:
    pin_dir.mkdir(parents=True, exist_ok=True)
    for name in ("model.pt", "replay.npz"):
        (pin_dir / name).write_bytes((run_dir / name).read_bytes())
    meta = {
        "label": label,
        "dir": pin_dir.name,
        "step": read_step(pin_dir / "model.pt"),
        "model_sha": sha16(pin_dir / "model.pt"),
        "replay_sha": sha16(pin_dir / "replay.npz"),
        "pinned_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (pin_dir / "pin.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def launch_window(repo: Path, run_dir: Path, flags: list[str], window_steps: int) -> None:
    cmd = [
        os.path.join(repo, ".venv-fceux/bin/python"), "-m", "mario_ai_fceux.apex_train",
        "--rom", "SuperMarioBros.nes", "--fceux", "fceux",
        "--run-dir", str(run_dir),
        "--steps", str(window_steps),
        "--checkpoint-every", "1000",
        "--window-layout", "config/fceux-window-layout.ini",
        "--cheats-enabled-workers", "",
        "--disable-eval",
        *flags,
    ]
    env = dict(os.environ, PYTHONPATH="python", LC_ALL="C")
    log = open("/tmp/drift-study.log", "a")
    proc = subprocess.Popen(cmd, cwd=str(repo), env=env,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    # The trainer exits gracefully when transitions_received >= window_steps.
    proc.wait()
    log.close()


def cpu_context() -> str:
    try:
        out = subprocess.run(["ps", "-Ao", "%cpu=,command="], capture_output=True, text=True).stdout
        busy = sum(float(l.split()[0]) for l in out.splitlines()
                   if ("fceux" in l or "apex_train" in l) and "ps -Ao" not in l)
        return f"{busy:.0f}% fceux+trainer CPU at eval start (0 = uncontaminated)"
    except Exception:
        return "cpu context unavailable"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--pin", type=Path, required=True,
                   help="Directory holding the t0 pin (model.pt + replay.npz).")
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--window-steps", type=int, default=25000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--eval-world", type=int, default=1)
    p.add_argument("--flags", type=str,
                   default="--worlds 1,1,1,1,2,2,2,2 --resume --frontier-spacing 128 "
                           "--frontier-retries 6 --learn-per-batch 4",
                   help="Training flags; kept identical across all windows (one change at a time).")
    args = p.parse_args()

    study = args.run_dir / "drift-study"
    study.mkdir(parents=True, exist_ok=True)
    flags = args.flags.split()
    points = []

    # Restore the t0 pin into the run dir so the study starts from it exactly.
    for name in ("model.pt", "replay.npz"):
        (args.run_dir / name).write_bytes((args.pin / name).read_bytes())
    points.append(pin_pair(args.run_dir, study / "pin-000-t0", "t0"))

    for w in range(1, args.windows + 1):
        launch_window(args.repo, args.run_dir, flags, args.window_steps)
        points.append(pin_pair(args.run_dir, study / f"pin-{w:03d}", f"window-{w}"))

    # One quiet-machine eval pass over every pin. REFUSE to start if actors are up:
    # contamination fakes wins and losses alike (project lesson).
    procs = subprocess.run(["pgrep", "-f", "apex_train|mario_ai_fceux.evaluate"],
                           capture_output=True, text=True).stdout.split()
    if procs:
        raise SystemExit(f"refusing to eval while {len(procs)} training/eval processes run: {procs}")
    ctx = cpu_context()

    for point in points:
        pin_dir = study / point["dir"]
        for name in ("model.pt", "replay.npz"):
            (args.run_dir / name).write_bytes((pin_dir / name).read_bytes())
        env = dict(os.environ, PYTHONPATH="python", LC_ALL="C")
        res = subprocess.run([
            os.path.join(args.repo, ".venv-fceux/bin/python"), "-m", "mario_ai_fceux.evaluate",
            "--rom", "SuperMarioBros.nes", "--fceux", "fceux",
            "--run-dir", str(args.run_dir), "--world", str(args.eval_world),
            "--episodes", str(args.eval_episodes), "--device", "cpu", "--trace-actions",
        ], cwd=str(args.repo), env=env, capture_output=True, text=True)
        # results.json lands in the run dir's evaluations; read the newest.
        evals = sorted((args.run_dir / "evaluations").glob("*/results.json"))
        r = json.loads(evals[-1].read_text()) if evals else {}
        xs = [e.get("max_x", 0) for e in r.get("episodes", [])]
        point.update({
            "wins": r.get("victories"),
            "episodes": r.get("episodes_finished"),
            "max_x_mean": round(sum(xs) / len(xs), 1) if xs else None,
            "max_x_median": sorted(xs)[len(xs) // 2] if xs else None,
            "cpu_context": ctx,
            "eval_log_sha": hashlib.sha256(res.stdout.encode()).hexdigest()[:16],
        })

    (study / "curve.json").write_text(json.dumps({
        "protocol": "windows pinned, one quiet pass, n=20 per point",
        "flags": args.flags,
        "cpu_context": ctx,
        "points": points,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"points": len(points), "out": str(study / "curve.json")}, indent=2))


if __name__ == "__main__":
    main()
