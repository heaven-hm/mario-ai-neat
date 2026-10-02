"""Environment assembly: level selection, wrapper stack, vectorization, device.

The PPO baseline is Super Mario Bros. 1 only. ``--world``/``--stage`` pick the
level, and World 1-1 is the default because it is the first promotion target.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium

from .compat import GymnasiumApiAdapter
from .rewards import MarioRewardWrapper, RewardConfig
from .rom import require_rom, resolve_rom, use_local_rom
from .synthetic import SyntheticScript, SyntheticSmb1Env
from .wrappers import (
    FRAME_SIZE,
    FRAME_SKIP,
    FRAME_STACK,
    FrameSkipMaxPool,
    FrameStack,
    Grayscale,
    Normalize,
    Resize,
)

ENVIRONMENT_SMB1 = "smb1"
ENVIRONMENT_SYNTHETIC = "synthetic"

DEFAULT_WORLD = 1
DEFAULT_STAGE = 1
DEFAULT_MAX_EPISODE_STEPS = 3000
DEFAULT_SEED = 2026


@dataclass(frozen=True)
class LevelSpec:
    """Which SMB1 level to train. SMB1 ships worlds 1-8 with four stages each."""

    world: int = DEFAULT_WORLD
    stage: int = DEFAULT_STAGE

    def __post_init__(self):
        if not 1 <= self.world <= 8:
            raise ValueError(f"world must be in 1..8, got {self.world}")
        if not 1 <= self.stage <= 4:
            raise ValueError(f"stage must be in 1..4, got {self.stage}")

    @property
    def label(self) -> str:
        return f"{self.world}-{self.stage}"

    @property
    def target(self) -> tuple[int, int]:
        return (self.world, self.stage)


@dataclass(frozen=True)
class EnvConfig:
    """Everything needed to build one wrapped environment."""

    level: LevelSpec = field(default_factory=LevelSpec)
    rom: Path | None = None
    environment: str = ENVIRONMENT_SMB1
    frame_skip: int = FRAME_SKIP
    frame_size: int = FRAME_SIZE
    frame_stack: int = FRAME_STACK
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS
    no_progress_frames: int = 0
    reward: RewardConfig = field(default_factory=RewardConfig)
    seed: int = DEFAULT_SEED
    synthetic: SyntheticScript = field(default_factory=SyntheticScript)

    def __post_init__(self):
        if self.environment not in (ENVIRONMENT_SMB1, ENVIRONMENT_SYNTHETIC):
            raise ValueError(f"environment must be {ENVIRONMENT_SMB1!r} or {ENVIRONMENT_SYNTHETIC!r}")
        if self.no_progress_frames < 0:
            raise ValueError("no_progress_frames must not be negative")

    @property
    def is_synthetic(self) -> bool:
        return self.environment == ENVIRONMENT_SYNTHETIC


class NoProgressLimit(gymnasium.Wrapper):
    """Truncate an episode that stops making ground.

    Off by default: waiting for a moving platform or a walking enemy is a real
    SMB1 strategy, so ending an episode for standing still is a choice, not a
    default. When enabled it truncates rather than terminates, so the value
    function bootstraps instead of treating a stall as a death.
    """

    def __init__(self, env, frames: int):
        super().__init__(env)
        if frames < 1:
            raise ValueError(f"no-progress window must be at least 1 frame, got {frames}")
        self.frames = frames
        self._best_x: int | None = None
        self._stalled = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._best_x = self._read(info)
        self._stalled = 0
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        current = self._read(info)
        if current is not None:
            if self._best_x is None or current > self._best_x:
                self._best_x = current
                self._stalled = 0
            else:
                self._stalled += self.frame_skip
            if self._stalled >= self.frames:
                truncated = True
        return observation, reward, terminated, truncated, info

    @property
    def frame_skip(self) -> int:
        return int(getattr(self.env, "skip", 1))

    @staticmethod
    def _read(info: dict) -> int | None:
        value = info.get("x_pos")
        try:
            return None if value is None else int(value)
        except (TypeError, ValueError):
            return None


class SeedOnReset(gymnasium.Wrapper):
    """Hand every episode's reset a fixed seed for reproducibility.

    Super Mario Bros. 1-1 is a fixed level and the emulator is deterministic,
    so this only pins the RNG used by any stochastic element in the stack; it
    is not what makes episodes differ, which is the policy's own sampling.
    """

    def __init__(self, env, seed: int):
        super().__init__(env)
        self.seed_value = int(seed)

    def reset(self, **kwargs):
        kwargs.setdefault("seed", self.seed_value)
        return self.env.reset(**kwargs)


def build_raw_environment(config: EnvConfig):
    """Build the unwrapped, adapter-free environment for a level."""
    if config.is_synthetic:
        return SyntheticSmb1Env(config.synthetic)
    require_rom(config.rom)
    use_local_rom(resolve_rom(config.rom))
    from gym_super_mario_bros.smb_env import SuperMarioBrosEnv

    return SuperMarioBrosEnv(rom_mode="vanilla", lost_levels=False, target=config.level.target)


def make_env(config: EnvConfig, rank: int = 0, record_trace: bool = False):
    """Return a thunk building one fully wrapped environment.

    ``record_trace`` is only used by evaluation, which keeps the full action
    trace; training keeps counters instead so worker memory stays flat.
    """
    from .actions import joypad_space

    def build():
        env = build_raw_environment(config)
        if not config.is_synthetic:
            env = joypad_space(env)
        env = GymnasiumApiAdapter(env)
        env = FrameSkipMaxPool(env, config.frame_skip)
        if config.no_progress_frames:
            env = NoProgressLimit(env, config.no_progress_frames)
        env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=config.max_episode_steps)
        env = MarioRewardWrapper(env, config.reward)
        env = Grayscale(env)
        env = Resize(env, config.frame_size)
        env = Normalize(env)
        env = FrameStack(env, config.frame_stack)
        env = SeedOnReset(env, config.seed + rank)
        return env

    return build


def start_method(workers: int) -> str | None:
    """Pick a safe start method for the worker count and platform.

    macOS defaults to ``spawn`` because forking a process that has already
    touched CUDA or MPS is unsafe; Linux uses ``forkserver``, which keeps a
    single interpreter start-up cost off every worker.
    """
    if workers <= 1:
        return None
    return "spawn" if platform.system() == "Darwin" else "forkserver"


def make_vec_env(
    config: EnvConfig,
    workers: int = 1,
    record_trace: bool = False,
    monitor: bool = False,
):
    """Build a vectorized environment with ``workers`` independent instances."""
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    thunks = [make_env(config, rank, record_trace) for rank in range(workers)]
    if workers == 1:
        return DummyVecEnv(thunks)
    method = start_method(workers)
    return SubprocVecEnv(thunks, start_method=method or "forkserver")


def resolve_device(requested: str | None = None) -> str:
    """Resolve ``auto`` to the best available device: CUDA, then MPS, then CPU."""
    import torch

    if requested and requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("--device mps was requested but MPS is not available")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_caveat(device: str) -> str | None:
    """Return a warning worth printing for the chosen device, if any."""
    if device == "mps":
        return (
            "MPS selected: Stable-Baselines3 documents MPS as inference-oriented, and "
            "PPO training on it can be slower or unstable than CPU for a CNN. Pass "
            "--device cpu if the run misbehaves."
        )
    return None


def describe_environment() -> dict[str, object]:
    """Record the exact library versions a run used."""
    description: dict[str, object] = {"python": platform.python_version(), "platform": platform.platform()}
    for name, module in (
        ("torch", "torch"),
        ("stable_baselines3", "stable_baselines3"),
        ("gymnasium", "gymnasium"),
        ("gym_super_mario_bros", "gym_super_mario_bros"),
        ("nes_py", "nes_py"),
        ("numpy", "numpy"),
        ("opencv", "cv2"),
    ):
        try:
            imported = __import__(module)
            description[name] = getattr(imported, "__version__", "unknown")
        except Exception as error:  # pragma: no cover - reporting path
            description[name] = f"unavailable: {type(error).__name__}"
    return description


def observation_shape(config: EnvConfig) -> tuple[int, int, int]:
    return (config.frame_stack, config.frame_size, config.frame_size)


__all__ = [
    "ENVIRONMENT_SMB1",
    "ENVIRONMENT_SYNTHETIC",
    "EnvConfig",
    "LevelSpec",
    "NoProgressLimit",
    "SeedOnReset",
    "build_raw_environment",
    "describe_environment",
    "device_caveat",
    "make_env",
    "make_vec_env",
    "observation_shape",
    "resolve_device",
    "start_method",
]
