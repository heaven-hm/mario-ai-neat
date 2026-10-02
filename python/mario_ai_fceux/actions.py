"""SMB1 high-level actions and their semi-Markov frame durations."""

from __future__ import annotations


# Each base action has short, normal, and committed horizons.  The normal
# 12-frame choices preserve the meaning of actions in older checkpoints.
ACTION_BASES = ("run", "jump_run", "retreat", "brake", "jump_place", "walk", "jump_back")
ACTION_DURATIONS = (6, 12, 24)
ACTION_COUNT = len(ACTION_BASES) * len(ACTION_DURATIONS)
LEGACY_ACTION_COUNT = 6
LEGACY_DURATION_FRAMES = 12
# If the network values two jump horizons almost equally, use the longer
# learned macro. A six-frame tap often cannot clear SMB1 obstacles reliably.
JUMP_HORIZON_TIE_TOLERANCE = 0.05
# At a clean SMB1 spawn Mario faces right on stable ground. Backtracking,
# braking, or jumping backward there cannot be a competent policy response.
SAFE_START_BASES = (0, 1, 4, 5)  # run, jump-run, jump-in-place, walk


def encode_action(base_index: int, duration_index: int) -> int:
    if not 0 <= base_index < len(ACTION_BASES):
        raise ValueError("invalid SMB1 action base")
    if not 0 <= duration_index < len(ACTION_DURATIONS):
        raise ValueError("invalid SMB1 action duration")
    return base_index * len(ACTION_DURATIONS) + duration_index


def decode_action(action: int) -> tuple[str, int]:
    if not 0 <= action < ACTION_COUNT:
        raise ValueError(f"invalid SMB1 action: {action}")
    base_index, duration_index = divmod(action, len(ACTION_DURATIONS))
    return ACTION_BASES[base_index], ACTION_DURATIONS[duration_index]


ACTION_NAMES = tuple(
    f"{base.replace('_', '+')}@{duration}"
    for base in ACTION_BASES
    for duration in ACTION_DURATIONS
)


def migrate_legacy_action(action: int) -> int:
    """Map a legacy fixed 12-frame action to its equivalent new action."""
    if not 0 <= action < LEGACY_ACTION_COUNT:
        raise ValueError(f"invalid legacy SMB1 action: {action}")
    return encode_action(action, ACTION_DURATIONS.index(LEGACY_DURATION_FRAMES))


def greedy_action(q_values) -> int:
    """Break near-ties toward a longer learned jump horizon.

    Old six-action checkpoints expand each learned value into three duration
    variants. Until training separates those values, ties favor the normal
    horizon for locomotion and the full horizon for jump actions.
    """
    values = [float(value) for value in q_values]
    if len(values) != ACTION_COUNT:
        raise ValueError(f"expected {ACTION_COUNT} Q values, got {len(values)}")
    best = max(values)
    tied = [index for index, value in enumerate(values) if best - value <= 1e-6]
    priorities = {}
    for base_index in range(len(ACTION_BASES)):
        preferred_duration = 2 if base_index in (1, 4, 6) else 1
        priorities[encode_action(base_index, preferred_duration)] = 1
    selected = max(tied, key=lambda index: (priorities.get(index, 0), -index))
    selected_base = selected // len(ACTION_DURATIONS)
    if selected_base in (1, 4, 6):
        candidates = [encode_action(selected_base, duration_index)
                      for duration_index in range(len(ACTION_DURATIONS))]
        near_tied = [index for index in candidates
                     if best - values[index] <= JUMP_HORIZON_TIE_TOLERANCE]
        if near_tied:
            selected = max(near_tied, key=lambda index: index % len(ACTION_DURATIONS))
    return selected


def safe_start_action(q_values) -> int:
    """Choose the best forward-capable action at a clear level start.

    This is a narrow safety constraint, not a replacement policy: it is used
    only before the first obstacle/gap, where moving left or braking has no
    valid SMB1 objective and a collapsed value estimate otherwise self-traps.
    """
    values = list(float(value) for value in q_values)
    if len(values) != ACTION_COUNT:
        raise ValueError(f"expected {ACTION_COUNT} Q values, got {len(values)}")
    masked = [value if index // len(ACTION_DURATIONS) in SAFE_START_BASES
              else float("-inf") for index, value in enumerate(values)]
    return greedy_action(masked)
