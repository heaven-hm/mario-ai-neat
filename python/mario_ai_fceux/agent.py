"""Double-DQN learner with dueling network, n-step returns, and replay."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import random
from pathlib import Path
from typing import Deque

import numpy as np
import torch
from torch import nn

from .model import DuelingQNetwork
from .replay import ReplayDatabase, Transition


@dataclass
class AgentConfig:
    observation_size: int = 184
    action_count: int = 6
    gamma: float = 0.99
    learning_rate: float = 2.5e-4
    batch_size: int = 128
    learning_starts: int = 5_000
    target_sync_steps: int = 2_000
    n_step: int = 3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 250_000
    guided_exploration_share: float = 0.90


class RainbowLiteAgent:
    """Practical Rainbow subset: double DQN, dueling, PER, and n-step returns."""

    def __init__(self, replay: ReplayDatabase, config: AgentConfig | None = None, device: str | None = None):
        self.replay = replay
        self.config = config or AgentConfig(observation_size=replay.observation_size)
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"))
        self.online = DuelingQNetwork(self.config.observation_size, self.config.action_count).to(self.device)
        self.target = DuelingQNetwork(self.config.observation_size, self.config.action_count).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.steps = 0
        self.optimizer_steps = 0
        self.pending: dict[str, Deque[Transition]] = {}

    @property
    def epsilon(self) -> float:
        fraction = min(1.0, self.steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + fraction * (self.config.epsilon_end - self.config.epsilon_start)

    def select_actions(self, states: np.ndarray, explore: bool = True) -> np.ndarray:
        return self.choose_actions(self.action_values(states), explore=explore)

    def safety_actions(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return a conservative SMB1 action prior and its immediate-danger mask.

        The prior is demonstration-free curriculum guidance, not a replacement
        for the Q network: it makes exploration useful by running right and
        holding jump+run when the RAM observation says an enemy is ahead.
        """
        states = np.asarray(states, dtype=np.float32).reshape(-1, self.config.observation_size)
        actions = np.zeros(len(states), dtype=np.int64)  # run right
        grounded = states[:, 171] > 0.0
        enemy_dx = states[:, 174]
        enemy_dy = states[:, 175]
        enemy_near = states[:, 183] > 0.0
        enemy_ahead = (enemy_dx > 0.02) & (enemy_dx < 0.42) & (enemy_dy > -0.55) & (enemy_dy < 0.55)
        danger = enemy_ahead | enemy_near
        # A held jump has to begin on the ground; while airborne we keep the
        # action instead of replacing it with a random direction mid-arc.
        actions[grounded & danger] = 1  # jump + run
        actions[(~grounded) & danger] = 1
        return actions, danger

    def choose_actions(self, scores: np.ndarray, states: np.ndarray | None = None,
                       explore: bool = True) -> np.ndarray:
        """Choose batched actions with safe, useful exploration for SMB1."""
        scores = np.asarray(scores, dtype=np.float32).reshape(-1, self.config.action_count)
        actions = scores.argmax(axis=1)
        safety_actions = danger = None
        if states is not None:
            safety_actions, danger = self.safety_actions(states)
        if explore:
            for index in range(len(actions)):
                if random.random() < self.epsilon:
                    # Uniform random exploration spent half its trials moving
                    # away from the goal or standing still.  Prefer an action
                    # that can produce a meaningful SMB1 trajectory, while a
                    # small random share still discovers alternatives.
                    if safety_actions is not None and random.random() < self.config.guided_exploration_share:
                        actions[index] = safety_actions[index]
                    else:
                        actions[index] = random.choice((0, 1, 4))
        # Never ask a newly trained, untrusted Q network to walk directly into
        # an immediately visible enemy.  The resulting successful jump
        # transitions enter replay and teach the network the same behaviour.
        if safety_actions is not None and danger is not None:
            actions[danger] = safety_actions[danger]
        return actions.astype(np.int64)

    def action_values(self, states: np.ndarray) -> np.ndarray:
        """Return the current online network's six action values for the HUD."""
        return self.inspect(states)[0]

    def inspect(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return Q-values and a 4x4 summary of the live 256-unit encoder."""
        states = np.asarray(states, dtype=np.float32).reshape(-1, self.config.observation_size)
        with torch.no_grad():
            tensor = torch.from_numpy(states).to(self.device)
            first_layer = self.online.encoder[1](self.online.encoder[0](tensor))
            encoded = self.online.encoder[3](self.online.encoder[2](first_layer))
            advantage = self.online.advantage(encoded)
            q_values = self.online.value(encoded) + advantage - advantage.mean(dim=1, keepdim=True)
            # Each square represents the mean activation of 16 encoder units.
            summary = encoded.reshape(-1, 16, 16).mean(dim=2)
        return q_values.cpu().numpy(), summary.cpu().numpy()

    def observe(self, worker_id: str, state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray, terminated: bool) -> None:
        """Add a worker transition and materialize mature n-step transitions."""
        pending = self.pending.setdefault(worker_id, deque())
        pending.append(Transition(np.asarray(state, dtype=np.float32), action, reward,
                                  np.asarray(next_state, dtype=np.float32), terminated, self.config.gamma))
        self.steps += 1
        self._drain_pending(worker_id, force=terminated)

    def _drain_pending(self, worker_id: str, force: bool) -> None:
        pending = self.pending[worker_id]
        while pending and (force or len(pending) >= self.config.n_step):
            accumulated_reward = 0.0
            discount = 1.0
            terminal = False
            next_state = pending[0].next_state
            for transition in list(pending)[:self.config.n_step]:
                accumulated_reward += discount * transition.reward
                discount *= self.config.gamma
                next_state = transition.next_state
                terminal = transition.terminated
                if terminal:
                    break
            first = pending.popleft()
            self.replay.add(Transition(first.state, first.action, accumulated_reward, next_state,
                                       terminal, 0.0 if terminal else discount, priority=1.0))
            if not force and len(pending) < self.config.n_step:
                break

    def learn(self) -> float | None:
        if len(self.replay) < max(self.config.learning_starts, self.config.batch_size):
            return None
        transition_ids, transitions, weights = self.replay.sample(self.config.batch_size)
        states = torch.as_tensor(np.stack([item.state for item in transitions]), device=self.device)
        actions = torch.as_tensor([item.action for item in transitions], device=self.device, dtype=torch.long)
        rewards = torch.as_tensor([item.reward for item in transitions], device=self.device)
        next_states = torch.as_tensor(np.stack([item.next_state for item in transitions]), device=self.device)
        terminated = torch.as_tensor([item.terminated for item in transitions], device=self.device, dtype=torch.bool)
        discounts = torch.as_tensor([item.discount for item in transitions], device=self.device)
        importance = torch.as_tensor(weights, device=self.device)
        current = self.online(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1)
            next_values = self.target(next_states).gather(1, next_actions[:, None]).squeeze(1)
            targets = rewards + (~terminated).float() * discounts * next_values
        td_error = targets - current
        loss = (importance * nn.functional.smooth_l1_loss(current, targets, reduction="none")).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.replay.update_priorities(transition_ids, td_error.detach().abs().cpu().numpy() + 1e-4)
        self.optimizer_steps += 1
        if self.optimizer_steps % self.config.target_sync_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach().cpu())

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.config), "steps": self.steps, "optimizer_steps": self.optimizer_steps,
                    "online": self.online.state_dict(), "target": self.target.state_dict(),
                    "optimizer": self.optimizer.state_dict()}, path)

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(payload["online"])
        self.target.load_state_dict(payload["target"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.steps = int(payload.get("steps", 0))
        self.optimizer_steps = int(payload.get("optimizer_steps", 0))
