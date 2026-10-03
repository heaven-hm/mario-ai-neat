"""Shared command-line configuration for the SMB1 PPO entry points."""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields, replace
from pathlib import Path

from .env import (
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_SEED,
    ENVIRONMENT_SMB1,
    ENVIRONMENT_SYNTHETIC,
    EnvConfig,
    LevelSpec,
)
from .rewards import RewardConfig
from .rom import require_rom, resolve_rom
from .wrappers import FRAME_SIZE, FRAME_SKIP, FRAME_STACK

# Reward flags accept None so "unset" keeps the RewardConfig default.
REWARD_FLAGS = {field.name: field.name.replace("_", "-") for field in fields(RewardConfig)}


def add_environment_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("environment")
    group.add_argument(
        "--env",
        dest="environment",
        choices=(ENVIRONMENT_SMB1, ENVIRONMENT_SYNTHETIC),
        default=ENVIRONMENT_SMB1,
        help="smb1 uses the real ROM environment; synthetic is a ROM-free stand-in for pipeline checks",
    )
    group.add_argument("--rom", type=Path, default=None, help="path to a locally supplied SMB1 ROM")
    group.add_argument("--world", type=int, default=1, help="SMB1 world, 1-8")
    group.add_argument("--stage", type=int, default=1, help="SMB1 stage within the world, 1-4")
    group.add_argument("--frame-skip", type=int, default=FRAME_SKIP)
    group.add_argument("--frame-size", type=int, default=FRAME_SIZE)
    group.add_argument("--frame-stack", type=int, default=FRAME_STACK)
    group.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    group.add_argument(
        "--no-progress-frames",
        type=int,
        default=0,
        help="truncate an episode after this many stalled frames (0 disables it)",
    )
    group.add_argument(
        "--render-mode",
        choices=("human", "rgb_array"),
        default=None,
        help="only the Gymnasium-native environment takes this at construction; used by watch",
    )


def add_reward_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("reward shaping")
    for name, flag in REWARD_FLAGS.items():
        group.add_argument(f"--{flag}", type=float, default=None, help=f"override RewardConfig.{name}")


def add_device_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("runtime")
    group.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or mps; auto prefers cuda, then mps, then cpu",
    )
    group.add_argument("--seed", type=int, default=DEFAULT_SEED)


def level_from(options: argparse.Namespace) -> LevelSpec:
    return LevelSpec(world=options.world, stage=options.stage)


def reward_config_from(options: argparse.Namespace) -> RewardConfig:
    overrides = {
        name: getattr(options, name) for name in REWARD_FLAGS if getattr(options, name, None) is not None
    }
    config = replace(RewardConfig(), **overrides) if overrides else RewardConfig()
    config.validate()
    return config


def env_config_from(options: argparse.Namespace) -> EnvConfig:
    config = EnvConfig(
        level=level_from(options),
        rom=resolve_rom(options.rom) if options.environment == ENVIRONMENT_SMB1 else None,
        environment=options.environment,
        frame_skip=options.frame_skip,
        frame_size=options.frame_size,
        frame_stack=options.frame_stack,
        max_episode_steps=options.max_episode_steps,
        no_progress_frames=options.no_progress_frames,
        reward=reward_config_from(options),
        seed=options.seed,
    )
    if not config.is_synthetic:
        require_rom(config.rom)
    return config


def environment_summary(config: EnvConfig) -> dict[str, object]:
    """Describe the environment for run metadata and evaluation reports."""
    summary: dict[str, object] = {
        "kind": config.environment,
        "level": config.level.label,
        "world": config.level.world,
        "stage": config.level.stage,
        "frame_skip": config.frame_skip,
        "frame_size": config.frame_size,
        "frame_stack": config.frame_stack,
        "max_episode_steps": config.max_episode_steps,
        "no_progress_frames": config.no_progress_frames,
        "reward": config.reward.__dict__,
    }
    if not config.is_synthetic:
        inspection = require_rom(config.rom)
        if not inspection.is_known_dump:
            print(
                f"warning: {inspection.path} is an SMB1-shaped image whose SHA-256 is not a "
                "known Super Mario Bros. (World) dump; the run will be recorded as unverified",
                file=sys.stderr,
            )
        summary["rom"] = {
            "path": inspection.path,
            "verdict": inspection.verdict,
            "dump_label": inspection.dump_label,
            "sha256": inspection.sha256,
            "md5": inspection.md5,
        }
        summary["cheats"] = "none: the environment is stock, with a user-supplied ROM"
    else:
        summary["rom"] = None
        summary["cheats"] = "not applicable: synthetic stand-in, not a benchmark environment"
    return summary
