"""Fixed seed suites for comparable Mario eval sweeps.

The champion gate compares candidate and incumbent on identical fixed suites.
Each suite is 20 episodes at one seed on 1-1; a promotion decision reports a
Wilson interval per suite and pools only suites whose evaluation conditions
match exactly (same ROM, FCEUX build, action repeat, start protocol, timing
regime). Sweeps run under the frame-deterministic evaluator once it lands;
until then legacy sweeps are timing-conditioned and never pooled with fixed
timing sweeps.
"""
from __future__ import annotations

from dataclasses import dataclass

EPISODES_PER_SUITE = 20


@dataclass(frozen=True)
class SeedSuite:
    name: str
    seed: int
    episodes: int = EPISODES_PER_SUITE
    world: int = 1
    level: int = 1


CHAMPION_SUITES = (
    SeedSuite("suite-a", 2026),
    SeedSuite("suite-b", 7),
    SeedSuite("suite-c", 424242),
)


def suite_by_name(name: str) -> SeedSuite:
    for suite in CHAMPION_SUITES:
        if suite.name == name:
            return suite
    raise KeyError(f"unknown seed suite: {name}")
