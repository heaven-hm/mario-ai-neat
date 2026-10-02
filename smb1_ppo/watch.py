"""Watch one rendered SMB1 PPO episode in a window.

    python -m smb1_ppo.watch --model runs/smb1-ppo-w1-1/best_model.zip

This is the only entry point that opens a window, it is never called
automatically, and it changes nothing about the measurement: it replays the
same greedy policy the evaluation uses. It needs a desktop session, because
nes-py's viewer is a pygame window that cannot open on a headless machine.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import cli
from .checkpoints import load_model
from .env import make_vec_env, resolve_device
from .rom import RomError
from .stats import RewardStatsVecEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m smb1_ppo.watch",
        description="Play a trained SMB1 PPO policy with the emulator window visible.",
    )
    cli.add_environment_arguments(parser)
    cli.add_reward_arguments(parser)
    cli.add_device_arguments(parser)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=0,
        help="stop an episode after this many decisions (0 runs it to its end)",
    )
    parser.add_argument("--frame-delay", type=float, default=0.0, help="seconds to sleep between decisions")
    # The Gymnasium-native stack needs this at construction time to open a window.
    parser.set_defaults(render_mode="human")
    return parser


def main() -> None:
    options = build_parser().parse_args()
    if options.episodes < 1:
        raise ValueError("--episodes must be positive")
    config = cli.env_config_from(options)
    if config.is_synthetic:
        raise SystemExit("the synthetic stand-in has no viewer; watch a real SMB1 ROM instead")
    device = resolve_device(options.device)
    environment = RewardStatsVecEnv(
        make_vec_env(config, workers=1, record_trace=True),
        config.level,
        frame_skip=config.frame_skip,
        record_trace=True,
        keep_records=True,
    )
    model_path = options.model.expanduser()
    if not model_path.is_file():
        raise SystemExit(f"model not found: {model_path}")
    model = load_model(model_path, env=environment, device=device)
    renderer = environment.venv.envs[0]

    try:
        for episode in range(1, options.episodes + 1):
            observations = environment.reset()
            finished = None
            seen = 0
            while finished is None:
                actions, _ = model.predict(observations, deterministic=True)
                observations, _, dones, _ = environment.step(actions)
                seen += 1
                if options.frame_delay:
                    time.sleep(options.frame_delay)
                try:
                    renderer.render()
                except Exception as error:
                    raise SystemExit(
                        f"rendering failed ({type(error).__name__}: {error}); "
                        "watch needs a desktop session with a display"
                    ) from error
                if dones[0]:
                    finished = environment.stream_records()[-1]
                elif options.max_decisions and seen >= options.max_decisions:
                    print(f"episode {episode}: stopped after {seen} decisions by --max-decisions")
                    break
            if finished is not None:
                print(
                    f"episode {episode}: won={finished.won} reason={finished.reason} "
                    f"max_x={finished.max_x} steps={finished.steps} "
                    f"duration={finished.duration_seconds:.1f}s"
                )
    except RomError as error:
        raise SystemExit(f"error: {error}") from error
    finally:
        environment.close()


if __name__ == "__main__":
    main()
