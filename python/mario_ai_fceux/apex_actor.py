"""Ape-X actor: runs FCEUX, collects experience batches, sends to shared queue.

Each actor plays independently with its own epsilon (Ape-X exploration schedule).
N-step returns are computed per-actor so trajectories are never mixed.
Actors send transition *batches* (default 32) to reduce IPC overhead.
Model weights are received from the learner and loaded periodically.
"""

from __future__ import annotations

import io
import logging
import os
import signal
import time
from collections import deque
from dataclasses import dataclass, replace
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Deque

import numpy as np
import torch

from .actions import ACTION_COUNT, LEGACY_DURATION_FRAMES, decode_action, greedy_action
from .environment import FileWorker, NoProgressTracker, Observation
from .model import RainbowNetwork
from .protocol import atomic_write_json, read_json
from .replay import Transition

logger = logging.getLogger(__name__)

# A death must outweigh several ordinary progress rewards.  The old -5
# terminal penalty was only a few maximum-sized (+2) progress decisions, so
# replay taught the policy that repeatedly reaching a dangerous state was
# still worthwhile.  Keep victory separate and strongly positive.
DEATH_REWARD_PENALTY = 20.0
VICTORY_REWARD_BONUS = 20.0

# Ape-X exploration schedule: actor 0 explores most, actor 7 exploits most.
# Matches the Ape-X paper's per-actor epsilon annealing philosophy.
APEX_EPSILONS = (0.40, 0.20, 0.10, 0.05, 0.025, 0.012, 0.006, 0.003)
UNSOLVED_WORLD_EPSILON_FLOOR = 0.10
FRONTIER_SPACING_PIXELS = 256
FRONTIER_RETRIES = 3


def _apex_epsilon(actor_index: int, total_actors: int) -> float:
    """Return actor-specific epsilon following the Ape-X schedule."""
    if total_actors <= 1:
        return 0.10
    fraction = actor_index / max(1, total_actors - 1)
    # Interpolate log-linearly between max and min epsilon.
    log_max = np.log(APEX_EPSILONS[0])
    log_min = np.log(APEX_EPSILONS[-1])
    return float(np.exp(log_max + fraction * (log_min - log_max)))


def training_epsilon(actor_index: int, total_actors: int, victories: int,
                     unsolved_floor: float = UNSOLVED_WORLD_EPSILON_FLOOR) -> float:
    """Give an unsolved assigned level meaningful exploration."""
    base = _apex_epsilon(actor_index, total_actors)
    return max(base, unsolved_floor) if victories == 0 else base


@dataclass
class ActorConfig:
    observation_size: int = 184
    action_count: int = ACTION_COUNT
    gamma: float = 0.99
    n_step: int = 3
    batch_size: int = 32          # transitions per queue push
    atom_count: int = 51
    value_min: float = -100.0
    value_max: float = 100.0
    weight_sync_every: int = 400  # steps between weight pulls
    seed: int = 7
    alternate_cheat_campaigns: bool = False
    unsolved_epsilon_floor: float = UNSOLVED_WORLD_EPSILON_FLOOR
    frontier_spacing: int = FRONTIER_SPACING_PIXELS
    frontier_retries: int = FRONTIER_RETRIES


class NStepBuffer:
    """Per-worker n-step return accumulator. Trajectories are never mixed."""

    def __init__(self, gamma: float, n_step: int) -> None:
        self.gamma = gamma
        self.n_step = n_step
        self.pending: Deque[Transition] = deque()

    def push(self, transition: Transition) -> list[Transition]:
        """Add one step; return ready n-step transitions."""
        self.pending.append(transition)
        return self._drain(force=transition.terminated)

    def flush(self) -> list[Transition]:
        """Force-drain remaining steps at episode end."""
        return self._drain(force=True)

    def _drain(self, force: bool) -> list[Transition]:
        ready: list[Transition] = []
        while self.pending and (force or len(self.pending) >= self.n_step):
            reward, discount, terminal = 0.0, 1.0, False
            next_state = self.pending[0].next_state
            for step in list(self.pending)[: self.n_step]:
                reward += discount * step.reward
                # Each decision carries its own duration-adjusted discount.
                discount *= step.discount
                next_state, terminal = step.next_state, step.terminated
                if terminal:
                    break
            first = self.pending.popleft()
            ready.append(
                Transition(
                    first.state,
                    first.action,
                    reward,
                    next_state,
                    terminal,
                    0.0 if terminal else discount,
                    priority=1.0,  # learner assigns real priority after TD-error
                )
            )
            if not force and len(self.pending) < self.n_step:
                break
        return ready


def _select_action(
    network: RainbowNetwork,
    support: torch.Tensor,
    state: np.ndarray,
    epsilon: float,
    action_count: int,
    device: torch.device,
) -> int:
    """Epsilon-greedy action selection using the actor's local network copy."""
    action, _, _ = _action_details(network, support, state, epsilon,
                                  action_count, device)
    return action


def _action_details(
    network: RainbowNetwork,
    support: torch.Tensor,
    state: np.ndarray,
    epsilon: float,
    action_count: int,
    device: torch.device,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Return action, Q values, and encoder summary for the live FCEUX HUD."""
    # Actor policy values use learned mean weights; exploration comes only
    # from the Ape-X epsilon schedule. NoisyNet stays active in the learner.
    network.eval()
    obs = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        encoded = network.encoder(obs)
        q_values = network(obs, support)
    greedy_index = (greedy_action(q_values[0].cpu().numpy())
                    if action_count == ACTION_COUNT
                    else int(q_values.argmax(dim=1).item()))
    if np.random.random() < epsilon:
        action = int(np.random.randint(action_count))
    else:
        action = greedy_index
    hidden_summary = encoded.reshape(-1, 16, 16).mean(dim=2)
    return action, q_values[0].cpu().numpy(), hidden_summary[0].cpu().numpy()


def _load_weights_from_bytes(network: RainbowNetwork, weight_bytes: bytes,
                              device: torch.device, evaluation: bool = False) -> None:
    """Deserialise a weight snapshot broadcast by the learner."""
    buf = io.BytesIO(weight_bytes)
    state_dict = torch.load(buf, map_location=device, weights_only=True)
    network.load_state_dict(state_dict)
    network.train(mode=not evaluation)


def _enqueue_batch(experience_queue: Queue, actor_index: int,
                   transitions: list[Transition]) -> None:
    """Send a complete batch, waiting for learner capacity instead of dropping it."""
    if transitions:
        experience_queue.put(("batch", actor_index, list(transitions)))


def _terminal_transition(previous: Observation, current: Observation,
                         action: int) -> Transition:
    """Create the final state-action-reward record for death or level completion."""
    return Transition(previous.state, action, _shaped_reward(previous, current),
                      current.state, True, 0.0)


def _publish_hud(worker: FileWorker, run_directory: str, observation: Observation,
                 action: int, q_values: np.ndarray, hidden: np.ndarray,
                 steps: int, episodes: int, deaths: int, victories: int,
                 best_x: int, epsilon: float) -> None:
    """Publish the actor's real state/action values for the FCEUX overlay."""
    learner = read_json(Path(run_directory) / "learner_status.json") or {}
    atomic_write_json(worker.directory / "hud.json", {
        "sequence": observation.sequence,
        "steps": steps,
        "updates": int(learner.get("optimizer_updates", 0)),
        "replay": int(learner.get("replay_transitions", 0)),
        "epsilon": epsilon,
        "episodes": episodes,
        "deaths": deaths,
        "victories": victories,
        "best_x": best_x,
        "loss": float(learner.get("latest_loss") or 0.0),
        "action": action,
        "values": [float(value) for value in q_values],
        "grid": [float(value) for value in observation.state[:169]],
        "globals": [float(value) for value in observation.state[169:184]],
        "hidden": [float(value) for value in hidden],
    })


def actor_main(
    actor_index: int,
    total_actors: int,
    worker: FileWorker,
    experience_queue: Queue,          # send batches of Transition to learner
    weight_queue: Queue,              # receive serialized weights from learner
    run_directory: str,
    config_dict: dict,
    device_str: str | None,
) -> None:
    """Entry point for one Ape-X actor subprocess.

    The actor:
    1. Plays Mario using a local copy of the network.
    2. Accumulates n-step returns without touching shared state.
    3. Pushes transition *batches* to experience_queue.
    4. Reloads weights from weight_queue every ``weight_sync_every`` steps.
    """
    # Ignore Ctrl-C; parent handles shutdown.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    config = ActorConfig(**config_dict)
    epsilon = training_epsilon(actor_index, total_actors, 0,
                               config.unsolved_epsilon_floor)
    device = torch.device(
        device_str or ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
    )
    support = torch.linspace(config.value_min, config.value_max,
                             config.atom_count, device=device)
    actor_seed = config.seed + actor_index * 1000
    np.random.seed(actor_seed)
    torch.manual_seed(actor_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actor_seed)
    network = RainbowNetwork(config.observation_size, config.action_count,
                             config.atom_count).to(device)
    # Train mode activates factorised NoisyNet exploration.  This network has
    # no dropout or batch-normalisation layers, so only NoisyLinear changes.
    # Pure epsilon-greedy actors: NoisyLinear uses learned mean weights in
    # eval mode. The learner keeps its separate network in train mode.
    network.eval()

    n_step_buf = NStepBuffer(config.gamma, config.n_step)
    progress_tracker = NoProgressTracker()
    batch: list[Transition] = []
    episode_replay: Deque[Transition] = deque(maxlen=8192)

    steps = 0
    last_sync = 0
    weights_ready = False
    episode_max_x = 0
    best_episode_x = 0
    episodes = 0
    victories = 0
    won_levels: set[tuple[int, int]] = set()
    levels_completed = 0
    campaigns_completed = 0
    frontier_course: tuple[int, int] | None = None
    next_frontier_x = 0
    frontier_available = False
    frontier_retries = 0
    frontier_resets = 0
    start_resets = 0

    metrics_path = Path(run_directory) / f"actor_{actor_index:02d}_metrics.json"

    def _flush_batch() -> None:
        if batch:
            # Backpressure preserves experience until the learner can consume it.
            _enqueue_batch(experience_queue, actor_index, batch)
            batch.clear()

    def _sync_weights() -> None:
        nonlocal last_sync, weights_ready
        # Drain all pending weight updates; use the most recent one.
        latest: bytes | None = None
        while True:
            try:
                _, weight_bytes = weight_queue.get_nowait()
                latest = weight_bytes
            except Exception:
                break
        if latest is not None:
            try:
                _load_weights_from_bytes(network, latest, device, evaluation=True)
                weights_ready = True
            except Exception as exc:
                logger.warning("actor %d weight load failed: %s", actor_index, exc)
        if weights_ready:
            last_sync = steps

    logger.info("Actor %d started (epsilon=%.4f)", actor_index, epsilon)

    while True:
        if not weights_ready or steps - last_sync >= config.weight_sync_every:
            _sync_weights()
        if not weights_ready:
            time.sleep(0.002)
            continue
        observation = worker.next_observation()
        if observation is None:
            time.sleep(0.002)
            # Still check weights while idle.
            if not weights_ready or steps - last_sync >= config.weight_sync_every:
                _sync_weights()
            continue
        if worker.consume_bridge_restart():
            # The terminal win was already flushed before the supervisor
            # restarted FCEUX. Do not create a transition across cheat modes.
            worker.previous = None
            n_step_buf.flush()
            episode_replay.clear()
            progress_tracker.reset()
            frontier_course = None
            frontier_available = False
            frontier_retries = 0

        course = (observation.world, observation.level)
        if course != frontier_course:
            # Each level owns its savestate. Never restore a checkpoint made
            # in a preceding level or after a worker's cheat-mode restart.
            frontier_course = course
            next_frontier_x = observation.world_x + config.frontier_spacing
            frontier_available = False
            frontier_retries = 0

        episode_max_x = max(episode_max_x, observation.world_x)
        best_episode_x = max(best_episode_x, episode_max_x)
        epsilon = training_epsilon(
            actor_index, total_actors,
            int((observation.world, observation.level) in won_levels),
            config.unsolved_epsilon_floor,
        )

        previous_duration = getattr(worker, "previous_action_duration", LEGACY_DURATION_FRAMES)
        if not observation.terminal and progress_tracker.update(
            observation.world_x, previous_duration if worker.previous is not None else 0
        ):
            observation = replace(observation, terminal=True, reason="stuck")

        if observation.terminal:
            # Replay must contain terminal outcomes, not just episode metrics.
            if worker.previous is not None:
                ready = n_step_buf.push(_terminal_transition(
                    worker.previous, observation, getattr(worker, "previous_action", 0),
                ))
                batch.extend(ready)
                episode_replay.extend(ready)
                steps += 1
            episodes += 1
            if observation.reason == "victory":
                victories += 1
                levels_completed += 1
                won_levels.add((observation.world, observation.level))
            # Flush n-step buffer at episode boundary.
            tail = n_step_buf.flush()
            batch.extend(tail)
            episode_replay.extend(tail)
            _flush_batch()
            if observation.reason == "victory" and episode_replay:
                # Rehearse the complete n-step winning trajectory after its
                # ordinary batch. The learner stores a bounded protected copy.
                experience_queue.put(("success", actor_index, observation.world,
                                      observation.level, list(episode_replay)))
            episode_replay.clear()
            if observation.reason == "victory":
                # Each actor owns one SMB1 world campaign.  The bridge keeps
                # its current-level checkpoint after 1-1/1-2/1-3 wins, then
                # restores that world's 1-1 state after 1-4.
                if observation.level >= 3:
                    campaigns_completed += 1
                    if config.alternate_cheat_campaigns:
                        # First campaign is powered; each later completed
                        # World-N-1..N-4 cycle flips to normal, then powered.
                        next_cheat_mode = campaigns_completed % 2 == 0
                        worker.restart_world_with_cheat_mode(observation, next_cheat_mode)
                    else:
                        worker.restart_world(observation)
                else:
                    worker.advance_level(observation)
            else:
                # Death and no-progress retries stay on the current level.
                restore_frontier = (frontier_available
                                    and frontier_retries < config.frontier_retries)
                worker.reset(observation, restore_frontier=restore_frontier)
                if restore_frontier:
                    frontier_retries += 1
                    frontier_resets += 1
                else:
                    # A few attempts from a saved frontier are useful for
                    # credit assignment; repeated failures return to the
                    # clean start so the policy cannot overfit one bad state.
                    frontier_retries = 0
                    start_resets += 1
            worker.previous = None
            worker.previous_action = 0  # type: ignore[attr-defined]
            worker.previous_action_duration = LEGACY_DURATION_FRAMES  # type: ignore[attr-defined]
            progress_tracker.reset()
            atomic_write_json(
                metrics_path,
                {
                    "actor": actor_index,
                    "epsilon": round(epsilon, 5),
                    "steps": steps,
                    "episodes": episodes,
                    "victories": victories,
                    "levels_completed": levels_completed,
                    "campaigns_completed": campaigns_completed,
                    "world": observation.world + 1,
                    "level": observation.level + 1,
                    "episode_max_x": episode_max_x,
                    "best_episode_x": best_episode_x,
                    "frontier_x": next_frontier_x - config.frontier_spacing
                    if frontier_available else None,
                    "frontier_retries": frontier_retries,
                    "frontier_resets": frontier_resets,
                    "start_resets": start_resets,
                },
            )
            episode_max_x = 0
            continue

        # Select action.
        action, q_values, hidden = _action_details(
            network, support, observation.state, epsilon, config.action_count, device,
        )
        _publish_hud(worker, run_directory, observation, action, q_values, hidden,
                     steps, episodes, episodes - victories, victories,
                     best_episode_x, epsilon)

        # Save only stable, grounded advances. A spatial interval avoids
        # writing a savestate every decision while providing a retryable
        # frontier for the next obstacle.
        checkpoint_frontier = (
            bool(observation.state[171])
            and observation.world_x >= next_frontier_x
        )
        if checkpoint_frontier:
            frontier_available = True
            frontier_retries = 0
            next_frontier_x = observation.world_x + config.frontier_spacing
        worker.send_action(observation, action, checkpoint_frontier=checkpoint_frontier)

        # Record transition once we have a previous state.
        if worker.previous is not None:
            prev = worker.previous
            prev_action = getattr(worker, "previous_action", 0)
            raw_reward = _shaped_reward(prev, observation, previous_duration)
            t = Transition(
                state=prev.state,
                action=prev_action,
                reward=raw_reward,
                next_state=observation.state,
                terminated=False,
                discount=config.gamma ** (
                    previous_duration
                    / LEGACY_DURATION_FRAMES
                ),
            )
            ready = n_step_buf.push(t)
            batch.extend(ready)
            episode_replay.extend(ready)
            steps += 1

            if len(batch) >= config.batch_size:
                _flush_batch()

        worker.previous = observation
        worker.previous_action = action  # type: ignore[attr-defined]
        worker.previous_action_duration = decode_action(action)[1]  # type: ignore[attr-defined]

        # Periodic weight sync.
        if steps - last_sync >= config.weight_sync_every:
            _sync_weights()


# ---------------------------------------------------------------------------
# Reward shaping (same formula as the old train.py, kept actor-local)
# ---------------------------------------------------------------------------

def _shaped_reward(previous: Observation, current: Observation,
                   duration_frames: int = LEGACY_DURATION_FRAMES) -> float:
    """Reward progress and charge game time so standing still loses value."""
    reward = max(-2.0, min(2.0, (current.world_x - previous.world_x) / 16.0))
    reward += max(-0.2, min(0.2, (current.power - previous.power) * 0.1))
    if current.terminal:
        reward += VICTORY_REWARD_BONUS if current.reason == "victory" else -DEATH_REWARD_PENALTY
    else:
        reward -= 0.04 * duration_frames / LEGACY_DURATION_FRAMES
    return reward
