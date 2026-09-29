"""Persistent prioritized replay storage backed by SQLite."""

from __future__ import annotations

from dataclasses import dataclass
import random
import sqlite3
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


class ReplayDatabase:
    """A bounded SQLite replay buffer with priority-biased candidate sampling.

    SQLite makes learning data restart-safe and inspectable. Sampling first takes
    a random candidate set in SQL, then applies proportional priorities in
    Python. It avoids loading the whole replay memory for every optimizer step.
    """

    def __init__(self, path: str | Path, observation_size: int, capacity: int = 250_000):
        self.path = Path(path)
        self.observation_size = observation_size
        self.capacity = capacity
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS transitions (
                id INTEGER PRIMARY KEY,
                state BLOB NOT NULL,
                action INTEGER NOT NULL,
                reward REAL NOT NULL,
                next_state BLOB NOT NULL,
                terminated INTEGER NOT NULL,
                discount REAL NOT NULL,
                priority REAL NOT NULL
            )"""
        )
        self.connection.execute("CREATE INDEX IF NOT EXISTS transition_priority ON transitions(priority)")
        self.connection.commit()

    def __len__(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM transitions").fetchone()[0])

    def close(self) -> None:
        self.connection.close()

    def add(self, transition: Transition) -> None:
        state = np.asarray(transition.state, dtype=np.float32).reshape(self.observation_size)
        next_state = np.asarray(transition.next_state, dtype=np.float32).reshape(self.observation_size)
        self.connection.execute(
            "INSERT INTO transitions(state,action,reward,next_state,terminated,discount,priority) VALUES(?,?,?,?,?,?,?)",
            (state.tobytes(), int(transition.action), float(transition.reward), next_state.tobytes(),
             int(transition.terminated), float(transition.discount), max(float(transition.priority), 1e-4)),
        )
        self.connection.commit()
        overflow = len(self) - self.capacity
        if overflow > 0:
            self.connection.execute(
                "DELETE FROM transitions WHERE id IN (SELECT id FROM transitions ORDER BY id ASC LIMIT ?)",
                (overflow,),
            )
            self.connection.commit()

    def sample(self, batch_size: int, candidate_multiplier: int = 8) -> tuple[list[int], list[Transition], np.ndarray]:
        count = len(self)
        if count < batch_size:
            raise ValueError("not enough replay transitions")
        candidate_size = min(count, max(batch_size, batch_size * candidate_multiplier))
        rows = self.connection.execute(
            "SELECT id,state,action,reward,next_state,terminated,discount,priority FROM transitions ORDER BY RANDOM() LIMIT ?",
            (candidate_size,),
        ).fetchall()
        priorities = np.asarray([max(row[7], 1e-4) for row in rows], dtype=np.float64)
        probabilities = priorities / priorities.sum()
        selected = np.random.choice(len(rows), size=batch_size, replace=False, p=probabilities)
        transitions: list[Transition] = []
        ids: list[int] = []
        for index in selected:
            row = rows[int(index)]
            ids.append(int(row[0]))
            transitions.append(
                Transition(
                    state=np.frombuffer(row[1], dtype=np.float32).copy(),
                    action=int(row[2]), reward=float(row[3]),
                    next_state=np.frombuffer(row[4], dtype=np.float32).copy(),
                    terminated=bool(row[5]), discount=float(row[6]), priority=float(row[7]),
                )
            )
        # Importance weights compensate for priority-biased sampling.
        selected_probabilities = probabilities[selected]
        weights = (len(rows) * selected_probabilities) ** -0.4
        weights = (weights / weights.max()).astype(np.float32)
        return ids, transitions, weights

    def update_priorities(self, transition_ids: list[int], priorities: np.ndarray) -> None:
        self.connection.executemany(
            "UPDATE transitions SET priority=? WHERE id=?",
            [(max(float(priority), 1e-4), transition_id) for transition_id, priority in zip(transition_ids, priorities)],
        )
        self.connection.commit()
