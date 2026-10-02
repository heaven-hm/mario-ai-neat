"""Bridge nes-py's legacy Gym API to the Gymnasium API used by wrappers and SB3.

The installed stack (``nes-py 8.2.1`` + ``gym-super-mario-bros 7.4.0``) speaks
the original Gym contract, which differs from Gymnasium in three ways that
matter here:

* ``reset(seed) -> observation`` instead of ``reset(...) -> (observation, info)``
* ``step(action) -> (observation, reward, done, info)`` instead of a five-tuple
* spaces are ``gym.spaces.Box`` / ``gym.spaces.Discrete``, and every Gymnasium
  wrapper and Stable-Baselines3 assert on the Gymnasium types

``GymnasiumApiAdapter`` is a Gymnasium ``Env`` that owns the legacy environment
and translates all three, so the rest of this package is written against one
contract and no shim dependency is needed.
"""

from __future__ import annotations

import inspect

import gymnasium
import numpy as np

# Carried across a legacy reset so the reward wrapper can take a first-step
# baseline without re-reading RAM itself.
STATE_KEYS = ("x_pos", "status", "life", "time", "flag_get")

# SMB1 RAM address gym-super-mario-bros reads for the vertical viewport.
VIEWPORT_ADDRESS = 0x00B5


def to_gymnasium_space(space):
    """Convert a legacy Gym space to its Gymnasium equivalent."""
    if isinstance(space, gymnasium.spaces.Space):
        return space
    try:
        import gym
    except ImportError:
        raise TypeError(
            f"cannot convert {type(space).__name__} to Gymnasium: the legacy gym package is not installed"
        ) from None

    if isinstance(space, gym.spaces.Box):
        return gymnasium.spaces.Box(low=np.asarray(space.low), high=np.asarray(space.high), dtype=space.dtype)
    if isinstance(space, gym.spaces.Discrete):
        return gymnasium.spaces.Discrete(space.n, start=int(getattr(space, "start", 0)))
    raise TypeError(f"cannot convert legacy space {type(space).__name__} to Gymnasium")


class GymnasiumApiAdapter(gymnasium.Env):
    """Expose a legacy Gym environment through the Gymnasium API.

    A Gymnasium-native environment (gym-super-mario-bros 9.x) already satisfies
    this contract, so the adapter becomes a pass-through for everything except
    space conversion, which it handles idempotently.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, env, render_mode: str = "human"):
        self.env = env
        self.observation_space = to_gymnasium_space(env.observation_space)
        self.action_space = to_gymnasium_space(env.action_space)
        self.render_mode: str | None = render_mode
        self._state: dict[str, object] = {}

    # MARK: Gymnasium API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        result = self._call_reset(seed, options)
        if isinstance(result, tuple):
            observation, info = result
            return observation, self._clean(info)
        return result, self._reset_state()

    def _call_reset(self, seed: int | None, options: dict | None):
        """Call the wrapped reset with only the arguments it actually accepts.

        The legacy stack is not uniform: ``nes_py``'s ``JoypadSpace`` overrides
        ``reset`` with no parameters at all, so it swallows a seed that the NES
        environment underneath would have accepted. Passing an unsupported
        keyword raises, so the signature decides. Super Mario Bros. 1-1 is a
        fixed level in a deterministic emulator, so a dropped seed does not
        change what an episode contains.
        """
        accepted = self._reset_parameters()
        keywords = {}
        if "seed" in accepted:
            keywords["seed"] = seed
        if "options" in accepted:
            keywords["options"] = options
        return self.env.reset(**keywords)

    def _reset_parameters(self) -> set[str]:
        try:
            parameters = inspect.signature(self.env.reset).parameters
        except (TypeError, ValueError):
            return set()
        if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return {"seed", "options"}
        return set(parameters)

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            return (
                observation,
                reward,
                bool(terminated),
                bool(truncated),
                self._clean(info),
            )
        observation, reward, done, info = result
        info = self._clean(info)
        truncated = bool(info.pop("TimeLimit.truncated", False))
        return observation, reward, bool(done) and not truncated, truncated, info

    def render(self):
        return self._call_render()

    def _call_render(self):
        """Render with the signature the wrapped environment exposes.

        The legacy NES environment takes ``render(mode=...)``; the Gymnasium one
        takes no arguments and reads its constructor's ``render_mode`` instead.
        """
        try:
            parameters = inspect.signature(self.env.render).parameters
        except (TypeError, ValueError):
            parameters = {}
        if parameters:
            return self.env.render(mode=self.render_mode or "human")
        return self.env.render()

    def close(self):
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    # MARK: State access

    @property
    def viewport(self) -> int | None:
        """Return SMB1's vertical viewport byte, or None when unreachable.

        Values above 1 mean Mario is below the visible screen, which is how
        gym-super-mario-bros itself detects a fall into a hole. The byte is not
        part of the public ``info`` dict, so it is probed separately and
        reported as unavailable instead of being inferred.
        """
        ram = getattr(self._base(), "ram", None)
        if ram is None:
            return None
        try:
            return int(ram[VIEWPORT_ADDRESS])
        except Exception:
            return None

    def _base(self):
        """Return the innermost environment, tolerating a plain Gym environment."""
        return getattr(self.env, "unwrapped", self.env)

    def _clean(self, info) -> dict[str, object]:
        """Copy info into plain Python values and remember the episode state."""
        normalized = dict(info) if isinstance(info, dict) else {}
        for key, value in tuple(normalized.items()):
            if isinstance(value, np.generic):
                normalized[key] = value.item()
        if any(key in normalized for key in STATE_KEYS):
            self._state = {key: normalized[key] for key in STATE_KEYS if key in normalized}
        return normalized

    def _reset_state(self) -> dict[str, object]:
        """Recover the state summary a legacy reset does not return.

        Upstream exposes the same accessor it uses to build step info, so
        reading it costs nothing and keeps the reset baseline identical to the
        first step's. When it is unavailable, callers treat the first step as
        the baseline, so this degrades to ``{}`` rather than to invented values.
        """
        if self._state:
            return dict(self._state)
        accessor = getattr(self._base(), "_get_info", None)
        if callable(accessor):
            try:
                return self._clean(accessor())
            except Exception:
                return {}
        return {}
