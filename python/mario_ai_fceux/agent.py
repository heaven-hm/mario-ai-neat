"""Complete Rainbow DQN: C51, Double DQN, dueling, NoisyNets, PER and n-step."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import os
import random
from pathlib import Path
from typing import Deque

import numpy as np
import torch
from torch import nn

from .model import RainbowNetwork
from .replay import PrioritizedReplayBuffer, Transition


@dataclass
class AgentConfig:
    observation_size: int = 184
    action_count: int = 6
    gamma: float = 0.99
    learning_rate: float = 6.25e-5
    batch_size: int = 128
    learning_starts: int = 10_000
    target_sync_steps: int = 2_000
    n_step: int = 3
    atom_count: int = 51
    value_min: float = -100.0
    value_max: float = 100.0
    per_beta_start: float = 0.4
    per_beta_steps: int = 1_000_000
    seed: int = 7


class RainbowAgent:
    """One reproducible full-Rainbow learner. It contains no Mario action prior."""

    def __init__(self, replay: PrioritizedReplayBuffer, config: AgentConfig | None = None,
                 device: str | None = None) -> None:
        self.replay = replay
        self.config = config or AgentConfig(observation_size=replay.observation_size)
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available()
                                              else "cuda" if torch.cuda.is_available() else "cpu"))
        self._set_seeds(self.config.seed)
        self.support = torch.linspace(self.config.value_min, self.config.value_max,
                                      self.config.atom_count, device=self.device)
        self.delta_z = (self.config.value_max - self.config.value_min) / (self.config.atom_count - 1)
        self.online = RainbowNetwork(self.config.observation_size, self.config.action_count,
                                     self.config.atom_count).to(self.device)
        self.target = RainbowNetwork(self.config.observation_size, self.config.action_count,
                                     self.config.atom_count).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.learning_rate, eps=1.5e-4)
        self.steps = 0
        self.optimizer_steps = 0
        self.pending: dict[str, Deque[Transition]] = {}

    @staticmethod
    def _set_seeds(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @property
    def per_beta(self) -> float:
        fraction = min(1.0, self.optimizer_steps / self.config.per_beta_steps)
        return self.config.per_beta_start + fraction * (1.0 - self.config.per_beta_start)

    def select_actions(self, states: np.ndarray, explore: bool = True) -> np.ndarray:
        """Choose actions from learned noisy Q-values; evaluation disables parameter noise."""
        states = np.asarray(states, dtype=np.float32).reshape(-1, self.config.observation_size)
        self.online.train(mode=explore)
        if explore:
            self.online.reset_noise()
        with torch.no_grad():
            actions = self.online(torch.from_numpy(states).to(self.device), self.support).argmax(dim=1)
        return actions.cpu().numpy().astype(np.int64)

    def inspect(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return expected Q-values and a small encoder summary for the FCEUX HUD."""
        states = np.asarray(states, dtype=np.float32).reshape(-1, self.config.observation_size)
        self.online.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(states).to(self.device)
            encoded = self.online.encoder(tensor)
            q_values = self.online(tensor, self.support)
            summary = encoded.reshape(-1, 16, 16).mean(dim=2)
        return q_values.cpu().numpy(), summary.cpu().numpy()

    def observe(self, worker_id: str, state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray, terminated: bool) -> None:
        pending = self.pending.setdefault(worker_id, deque())
        pending.append(Transition(np.asarray(state, dtype=np.float32), action, reward,
                                  np.asarray(next_state, dtype=np.float32), terminated, self.config.gamma))
        self.steps += 1
        self._drain_pending(worker_id, force=terminated)

    def _drain_pending(self, worker_id: str, force: bool) -> None:
        pending = self.pending[worker_id]
        while pending and (force or len(pending) >= self.config.n_step):
            reward, discount, terminal = 0.0, 1.0, False
            next_state = pending[0].next_state
            for transition in list(pending)[:self.config.n_step]:
                reward += discount * transition.reward
                discount *= self.config.gamma
                next_state, terminal = transition.next_state, transition.terminated
                if terminal:
                    break
            first = pending.popleft()
            self.replay.add(Transition(first.state, first.action, reward, next_state, terminal,
                                       0.0 if terminal else discount, self.replay.max_priority))
            if not force and len(pending) < self.config.n_step:
                break

    def _project_distribution(self, next_distribution: torch.Tensor, rewards: torch.Tensor,
                              discounts: torch.Tensor, terminated: torch.Tensor) -> torch.Tensor:
        batch_size = rewards.shape[0]
        target_values = rewards[:, None] + (~terminated).float()[:, None] * discounts[:, None] * self.support
        target_values = target_values.clamp(self.config.value_min, self.config.value_max)
        positions = (target_values - self.config.value_min) / self.delta_z
        lower, upper = positions.floor().long(), positions.ceil().long()
        projected = torch.zeros(batch_size, self.config.atom_count, device=self.device)
        offset = (torch.arange(batch_size, device=self.device) * self.config.atom_count).unsqueeze(1)
        projected.view(-1).index_add_(0, (lower + offset).view(-1),
                                      (next_distribution * (upper.float() - positions)).view(-1))
        projected.view(-1).index_add_(0, (upper + offset).view(-1),
                                      (next_distribution * (positions - lower.float())).view(-1))
        # When an atom lands exactly on a support point lower == upper, retain its mass.
        exact = lower == upper
        projected.view(-1).index_add_(0, (lower + offset)[exact], next_distribution[exact])
        return projected

    def learn(self) -> float | None:
        if len(self.replay) < max(self.config.learning_starts, self.config.batch_size):
            return None
        indices, transitions, importance_weights = self.replay.sample(self.config.batch_size, self.per_beta)
        states = torch.as_tensor(np.stack([item.state for item in transitions]), device=self.device)
        actions = torch.as_tensor([item.action for item in transitions], device=self.device, dtype=torch.long)
        rewards = torch.as_tensor([item.reward for item in transitions], device=self.device)
        next_states = torch.as_tensor(np.stack([item.next_state for item in transitions]), device=self.device)
        terminated = torch.as_tensor([item.terminated for item in transitions], device=self.device, dtype=torch.bool)
        discounts = torch.as_tensor([item.discount for item in transitions], device=self.device)
        importance = torch.as_tensor(importance_weights, device=self.device)
        self.online.train()
        self.online.reset_noise()
        self.target.reset_noise()
        log_probabilities = self.online.distribution(states)[torch.arange(len(actions), device=self.device), actions].log()
        with torch.no_grad():
            next_actions = self.online(next_states, self.support).argmax(dim=1)
            next_distribution = self.target.distribution(next_states)[torch.arange(len(actions), device=self.device), next_actions]
            target_distribution = self._project_distribution(next_distribution, rewards, discounts, terminated)
        per_item_loss = -(target_distribution * log_probabilities).sum(dim=1)
        loss = (importance * per_item_loss).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.replay.update_priorities(indices, per_item_loss.detach().cpu().numpy() + self.replay.priority_epsilon)
        self.optimizer_steps += 1
        if self.optimizer_steps % self.config.target_sync_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach().cpu())

    def save(self, path: str | Path, replay_path: str | Path | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            torch.save({"algorithm": "Rainbow DQN (C51 + NoisyNet + Double + Dueling + PER + n-step)",
                        "config": asdict(self.config), "steps": self.steps, "optimizer_steps": self.optimizer_steps,
                        "online": self.online.state_dict(), "target": self.target.state_dict(),
                        "optimizer": self.optimizer.state_dict(), "python_random": random.getstate(),
                        "numpy_random": np.random.get_state(), "torch_random": torch.get_rng_state()}, temporary)
            os.replace(temporary, path)
            if replay_path is not None:
                self.replay.save(replay_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(payload["online"])
        self.target.load_state_dict(payload["target"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.steps, self.optimizer_steps = int(payload.get("steps", 0)), int(payload.get("optimizer_steps", 0))
        if "python_random" in payload:
            random.setstate(payload["python_random"])
            np.random.set_state(payload["numpy_random"])
            torch_state = payload["torch_random"]
            # Checkpoints created before the reproducibility format may carry
            # a non-Tensor value here. Keep their trained weights, but do not
            # crash a resume over an unusable old RNG record.
            if isinstance(torch_state, torch.Tensor) and torch_state.dtype == torch.uint8:
                torch.set_rng_state(torch_state)


# Old import name remains available for external users; it now implements full Rainbow.
RainbowLiteAgent = RainbowAgent
