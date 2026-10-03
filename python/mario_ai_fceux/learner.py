"""Dedicated Rainbow learner process; collectors never touch replay or optimizer state."""

from __future__ import annotations

from multiprocessing.queues import Queue
from pathlib import Path
import queue
import time
from typing import Any

import numpy as np

from .agent import AgentConfig, RainbowAgent
from .replay import PrioritizedReplayBuffer


def learner_main(inbox: Queue, outbox: Queue, run_directory: str, config_dict: dict[str, Any],
                 resume: bool, device: str | None) -> None:
    """Own replay, optimizer, RNG state, and checkpoints in one process."""
    directory = Path(run_directory)
    config = AgentConfig(**config_dict)
    replay_path, checkpoint_path = directory / "replay.npz", directory / "model.pt"
    if resume and checkpoint_path.exists():
        checkpoint_to_load, replay = RainbowAgent.load_checkpoint_pair(
            checkpoint_path, replay_path, config.observation_size, config.seed,
        )
    elif resume and replay_path.exists():
        checkpoint_to_load = checkpoint_path
        replay = PrioritizedReplayBuffer.load(replay_path, config.seed)
    else:
        checkpoint_to_load = checkpoint_path
        replay = PrioritizedReplayBuffer(config.observation_size, seed=config.seed)
    agent = RainbowAgent(replay, config=config, device=device)
    if resume and checkpoint_path.exists():
        try:
            agent.load(checkpoint_to_load)
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"Could not safely resume {checkpoint_path}; refusing to discard learned state: {error}"
            ) from error
    last_loss: float | None = None
    last_save_steps = agent.steps
    active = True
    while active:
        try:
            message = inbox.get(timeout=0.05)
        except queue.Empty:
            message = None
        if message is not None:
            kind = message[0]
            if kind == "transition":
                _, worker_id, state, action, reward, next_state, terminal = message
                agent.observe(worker_id, state, int(action), float(reward), next_state, bool(terminal))
                # Learn independently of the collectors; drain extra transitions quickly.
                for _ in range(2):
                    learned = agent.learn()
                    if learned is not None:
                        last_loss = learned
            elif kind == "act":
                _, request_id, states, explore = message
                state_array = np.asarray(states, dtype=np.float32)
                values, hidden = agent.inspect(state_array)
                actions = agent.select_actions(state_array, explore=bool(explore))
                outbox.put(("act", request_id, actions, values, hidden, {
                    "steps": agent.steps, "optimizer_updates": agent.optimizer_steps,
                    "replay_transitions": len(replay), "epsilon": 0.0, "latest_loss": last_loss,
                }))
            elif kind == "status":
                outbox.put(("status", {"steps": agent.steps, "optimizer_updates": agent.optimizer_steps,
                                        "replay_transitions": len(replay), "epsilon": 0.0,
                                        "latest_loss": last_loss}))
            elif kind == "save":
                agent.save(checkpoint_path, replay_path)
            elif kind == "stop":
                active = False
        if agent.steps - last_save_steps >= 25_000:
            agent.save(checkpoint_path, replay_path)
            last_save_steps = agent.steps
    agent.save(checkpoint_path, replay_path)
