"""Fast in-memory, globally prioritized experience replay for Rainbow DQN.

The buffer deliberately stays out of SQLite's hot path. FCEUX collectors send
transitions to one learner, which keeps this array-backed buffer in RAM. A
compressed snapshot is written only with a training checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    discount: float
    priority: float = 1.0


class SumTree:
    """Binary tree supporting exact global proportional-priority sampling."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.values = np.zeros(2 * capacity, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.values[1])

    def update(self, index: int, value: float) -> None:
        node = index + self.capacity
        difference = value - self.values[node]
        self.values[node] = value
        node //= 2
        while node:
            self.values[node] += difference
            node //= 2

    def find_prefixsum(self, value: float) -> int:
        node = 1
        while node < self.capacity:
            left = node * 2
            if value <= self.values[left]:
                node = left
            else:
                value -= self.values[left]
                node = left + 1
        return node - self.capacity


class PrioritizedReplayBuffer:
    """Array-backed replay with true global PER and no database transactions."""

    def __init__(self, observation_size: int, capacity: int = 100_000,
                 alpha: float = 0.6, priority_epsilon: float = 1e-5,
                 seed: int = 0) -> None:
        self.observation_size = observation_size
        self.capacity = capacity
        self.alpha = alpha
        self.priority_epsilon = priority_epsilon
        self.rng = np.random.default_rng(seed)
        self.states = np.zeros((capacity, observation_size), dtype=np.float32)
        self.next_states = np.zeros((capacity, observation_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int16)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.discounts = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.bool_)
        self.tree = SumTree(capacity)
        self.size = 0
        self.position = 0
        self.max_priority = 1.0

    def __len__(self) -> int:
        return self.size

    def add(self, transition: Transition) -> None:
        index = self.position
        self.states[index] = np.asarray(transition.state, dtype=np.float32).reshape(self.observation_size)
        self.next_states[index] = np.asarray(transition.next_state, dtype=np.float32).reshape(self.observation_size)
        self.actions[index] = int(transition.action)
        self.rewards[index] = float(transition.reward)
        self.discounts[index] = float(transition.discount)
        self.terminated[index] = bool(transition.terminated)
        priority = max(float(transition.priority), self.max_priority, self.priority_epsilon)
        self.tree.update(index, priority ** self.alpha)
        self.max_priority = max(self.max_priority, priority)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float) -> tuple[np.ndarray, list[Transition], np.ndarray]:
        if self.size < batch_size:
            raise ValueError("not enough replay transitions")
        total = self.tree.total
        if total <= 0:
            raise RuntimeError("replay priorities are empty")
        boundaries = np.linspace(0.0, total, batch_size + 1)
        samples = self.rng.uniform(boundaries[:-1], boundaries[1:])
        indices = np.asarray([self.tree.find_prefixsum(float(sample)) for sample in samples], dtype=np.int64)
        priorities = self.tree.values[indices + self.capacity]
        probabilities = priorities / total
        weights = (self.size * probabilities) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)
        transitions = [Transition(self.states[index].copy(), int(self.actions[index]), float(self.rewards[index]),
                                  self.next_states[index].copy(), bool(self.terminated[index]),
                                  float(self.discounts[index]), float(priorities[row]))
                       for row, index in enumerate(indices)]
        return indices, transitions, weights

    def update_priorities(self, indices: np.ndarray | list[int], priorities: np.ndarray) -> None:
        for index, priority in zip(indices, priorities):
            raw_priority = max(float(priority), self.priority_epsilon)
            self.tree.update(int(index), raw_priority ** self.alpha)
            self.max_priority = max(self.max_priority, raw_priority)

    def save(self, path: str | Path) -> None:
        """Atomically persist all replay state needed to resume sampling exactly."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, observation_size=self.observation_size, capacity=self.capacity,
                                alpha=self.alpha, priority_epsilon=self.priority_epsilon, size=self.size,
                                position=self.position, max_priority=self.max_priority, states=self.states[:self.size],
                                next_states=self.next_states[:self.size], actions=self.actions[:self.size],
                                rewards=self.rewards[:self.size], discounts=self.discounts[:self.size],
                                terminated=self.terminated[:self.size], tree=self.tree.values,
                                rng_state=np.asarray([self.rng.bit_generator.state], dtype=object))
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path, seed: int = 0) -> "PrioritizedReplayBuffer":
        with np.load(Path(path), allow_pickle=True) as payload:
            buffer = cls(int(payload["observation_size"]), int(payload["capacity"]), float(payload["alpha"]),
                         float(payload["priority_epsilon"]), seed)
            buffer.size = int(payload["size"])
            buffer.position = int(payload["position"])
            buffer.max_priority = float(payload["max_priority"])
            buffer.states[:buffer.size] = payload["states"]
            buffer.next_states[:buffer.size] = payload["next_states"]
            buffer.actions[:buffer.size] = payload["actions"]
            buffer.rewards[:buffer.size] = payload["rewards"]
            buffer.discounts[:buffer.size] = payload["discounts"]
            buffer.terminated[:buffer.size] = payload["terminated"]
            buffer.tree.values[:] = payload["tree"]
            buffer.rng.bit_generator.state = payload["rng_state"][0]
        return buffer


# Compatibility alias for code that imported the old SQLite-backed class.
ReplayDatabase = PrioritizedReplayBuffer
