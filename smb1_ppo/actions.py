"""The small discrete SMB1 action space used by the PPO baseline.

The Ape-X Rainbow system compresses each button press into a base action and a
frame duration, giving 7 bases x 3 durations = 21 macro actions. PPO does not
need that: frame skip and the CNN policy resolve timing themselves, so this
package uses one press per decision and a flat table of seven actions.

Action indices are part of the checkpoint contract. Never reorder this tuple.
"""

from __future__ import annotations

from collections import Counter

# One entry per discrete action: the NES buttons held for the whole decision.
# "right" and "left" are mutually exclusive in SMB1, so no entry mixes them.
ACTION_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("NOOP", ()),
    ("right", ("right",)),
    ("right+A", ("right", "A")),
    ("right+B", ("right", "B")),
    ("right+A+B", ("right", "A", "B")),
    ("left", ("left",)),
    ("left+A", ("left", "A")),
)

MAX_ACTIONS = 7

# The layout nes-py's JoypadSpace expects: one button-name list per action.
ACTIONS: tuple[list[str], ...] = tuple(list(buttons) for _, buttons in ACTION_SPECS)

ACTION_NAMES: tuple[str, ...] = tuple(name for name, _ in ACTION_SPECS)

ACTION_COUNT = len(ACTION_SPECS)

if ACTION_COUNT > MAX_ACTIONS:  # pragma: no cover - guards a source edit
    raise AssertionError(f"SMB1 PPO allows at most {MAX_ACTIONS} actions, found {ACTION_COUNT}")


def action_name(action: int) -> str:
    """Return the documented name of an action index."""
    return ACTION_NAMES[validate_action(action)]


def validate_action(action: int) -> int:
    """Return the action unchanged, or raise if it is outside the table."""
    if not isinstance(action, int) or isinstance(action, bool):
        raise TypeError(f"action must be an int, got {type(action).__name__}")
    if not 0 <= action < ACTION_COUNT:
        raise ValueError(f"action must be in 0..{ACTION_COUNT - 1}, got {action}")
    return action


def action_counts(actions) -> list[int]:
    """Count how often each action appeared in a trace, ready for CSV output."""
    counted = Counter(validate_action(int(action)) for action in actions)
    return [counted.get(index, 0) for index in range(ACTION_COUNT)]


def buttons_for(action: int) -> list[str]:
    """Return a copy of the button list for an action index."""
    return list(ACTIONS[validate_action(action)])


def joypad_space(env):
    """Wrap a raw NES environment in nes-py's discrete action adapter."""
    from nes_py.wrappers import JoypadSpace

    return JoypadSpace(env, list(ACTIONS))
