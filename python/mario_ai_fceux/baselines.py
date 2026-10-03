"""Compact, deliberately non-Rainbow DDQN and PPO baselines for SMB1.

These implementations share Rainbow's 184-value observation and six-action
contract, but use their standard algorithmic updates so benchmark results are
meaningful rather than renamed Rainbow checkpoints.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def _atomic_torch_save(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass
class BaselineConfig:
    observation_size: int = 184
    action_count: int = 6
    hidden_size: int = 256
    gamma: float = 0.99
    learning_rate: float = 2.5e-4
    batch_size: int = 128
    replay_capacity: int = 100_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 250_000
    target_sync_steps: int = 2_000
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 128
    gae_lambda: float = 0.95
    seed: int = 7


class BasicQNetwork(nn.Module):
    """Ordinary MLP Q function, without C51, NoisyNet, or dueling heads."""

    def __init__(self, observation_size: int, hidden_size: int, action_count: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.layers(observations)


class ActorCriticNetwork(nn.Module):
    """Shared MLP with a categorical actor and scalar critic for PPO."""

    def __init__(self, observation_size: int, hidden_size: int, action_count: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observation_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_count)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(observations)
        return self.actor(encoded), self.critic(encoded).squeeze(-1)


class DDQNAgent:
    """Double DQN with uniform in-memory replay and epsilon-greedy exploration."""

    algorithm = "Basic DDQN (Double DQN, uniform replay)"

    def __init__(self, config: BaselineConfig | None = None, device: str | None = None) -> None:
        self.config = config or BaselineConfig()
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available()
                                              else "cuda" if torch.cuda.is_available() else "cpu"))
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        self.random = random.Random(self.config.seed)
        self.online = BasicQNetwork(self.config.observation_size, self.config.hidden_size,
                                    self.config.action_count).to(self.device)
        self.target = BasicQNetwork(self.config.observation_size, self.config.hidden_size,
                                    self.config.action_count).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.replay: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=self.config.replay_capacity)
        self.steps = 0
        self.updates = 0

    @property
    def epsilon(self) -> float:
        progress = min(1.0, self.steps / max(1, self.config.epsilon_decay_steps))
        return self.config.epsilon_start + progress * (self.config.epsilon_end - self.config.epsilon_start)

    def select_actions(self, states: np.ndarray, explore: bool = False) -> np.ndarray:
        observations = np.asarray(states, dtype=np.float32).reshape(-1, self.config.observation_size)
        with torch.no_grad():
            values = self.online(torch.as_tensor(observations, device=self.device))
            actions = values.argmax(dim=1).cpu().numpy()
        if explore:
            for index in range(len(actions)):
                if self.random.random() < self.epsilon:
                    actions[index] = self.random.randrange(self.config.action_count)
        return actions.astype(np.int64)

    def observe(self, state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray, terminated: bool) -> None:
        self.replay.append((np.asarray(state, dtype=np.float32).copy(), int(action), float(reward),
                            np.asarray(next_state, dtype=np.float32).copy(), bool(terminated)))
        self.steps += 1

    def learn(self) -> float | None:
        if len(self.replay) < self.config.batch_size:
            return None
        batch = self.random.sample(self.replay, self.config.batch_size)
        states, actions, rewards, next_states, terminals = zip(*batch)
        state_tensor = torch.as_tensor(np.stack(states), device=self.device)
        action_tensor = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        reward_tensor = torch.as_tensor(rewards, device=self.device)
        next_tensor = torch.as_tensor(np.stack(next_states), device=self.device)
        terminal_tensor = torch.as_tensor(terminals, device=self.device, dtype=torch.bool)
        current = self.online(state_tensor).gather(1, action_tensor[:, None]).squeeze(1)
        with torch.no_grad():
            next_actions = self.online(next_tensor).argmax(dim=1)
            next_values = self.target(next_tensor).gather(1, next_actions[:, None]).squeeze(1)
            targets = reward_tensor + (~terminal_tensor).float() * self.config.gamma * next_values
        loss = nn.functional.smooth_l1_loss(current, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_sync_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach().cpu())

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        _atomic_torch_save({"algorithm": self.algorithm, "config": asdict(self.config),
                            "online": self.online.state_dict(), "target": self.target.state_dict(),
                            "optimizer": self.optimizer.state_dict(), "steps": self.steps,
                            "updates": self.updates, "replay": list(self.replay),
                            "python_random": self.random.getstate(),
                            "numpy_random": np.random.get_state(),
                            "torch_random": torch.get_rng_state()}, destination)

    def load(self, path: str | Path) -> None:
        # Keep RNG tensors on CPU even when model weights load to MPS/CUDA.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("algorithm") != self.algorithm or payload.get("config") != asdict(self.config):
            raise ValueError("DDQN checkpoint algorithm/configuration mismatch")
        self.online.load_state_dict(payload["online"])
        self.target.load_state_dict(payload["target"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.steps, self.updates = int(payload["steps"]), int(payload["updates"])
        self.replay = deque(payload.get("replay", ()), maxlen=self.config.replay_capacity)
        if "python_random" in payload:
            self.random.setstate(payload["python_random"])
        if isinstance(payload.get("numpy_random"), tuple):
            np.random.set_state(payload["numpy_random"])
        if isinstance(payload.get("torch_random"), torch.Tensor):
            torch.set_rng_state(payload["torch_random"])


class PPOAgent:
    """Clipped categorical PPO with generalized advantage estimation."""

    algorithm = "PPO (clipped categorical policy, GAE)"

    def __init__(self, config: BaselineConfig | None = None, device: str | None = None) -> None:
        self.config = config or BaselineConfig()
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available()
                                              else "cuda" if torch.cuda.is_available() else "cpu"))
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        self.random = np.random.default_rng(self.config.seed)
        self.policy = ActorCriticNetwork(self.config.observation_size, self.config.hidden_size,
                                         self.config.action_count).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        self.rollout: dict[str, list[tuple[np.ndarray, int, float, float, float, bool]]] = {}
        self.steps = 0
        self.updates = 0

    def select_actions(self, states: np.ndarray, explore: bool = False) -> np.ndarray:
        observations = np.asarray(states, dtype=np.float32).reshape(-1, self.config.observation_size)
        with torch.no_grad():
            logits, _ = self.policy(torch.as_tensor(observations, device=self.device))
        if explore:
            return Categorical(logits=logits).sample().cpu().numpy().astype(np.int64)
        return logits.argmax(dim=1).cpu().numpy().astype(np.int64)

    def act_with_statistics(self, state: np.ndarray) -> tuple[int, float, float]:
        observation = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.policy(observation)
            distribution = Categorical(logits=logits)
            action = distribution.sample()
        return int(action.item()), float(distribution.log_prob(action).item()), float(value.item())

    def value(self, state: np.ndarray) -> float:
        observation = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, value = self.policy(observation)
        return float(value.item())

    def observe(self, worker_id: str, state: np.ndarray, action: int, reward: float,
                old_log_probability: float, value: float, terminated: bool) -> None:
        trajectory = self.rollout.setdefault(worker_id, [])
        trajectory.append((np.asarray(state, dtype=np.float32).copy(), int(action), float(reward),
                           float(old_log_probability), float(value), bool(terminated)))
        self.steps += 1

    def learn(self, bootstrap_values: dict[str, float] | None = None) -> float | None:
        if not any(self.rollout.values()):
            return None
        bootstrap_values = bootstrap_values or {}
        entries: list[tuple[np.ndarray, int, float, float, float, bool]] = []
        advantages_parts: list[np.ndarray] = []
        returns_parts: list[np.ndarray] = []
        for worker_id, trajectory in self.rollout.items():
            trajectory_advantages = np.zeros(len(trajectory), dtype=np.float32)
            gae = 0.0
            next_value = float(bootstrap_values.get(worker_id, 0.0))
            for index in range(len(trajectory) - 1, -1, -1):
                _, _, reward, _, value, terminated = trajectory[index]
                continuation = 0.0 if terminated else 1.0
                delta = reward + self.config.gamma * next_value * continuation - value
                gae = delta + self.config.gamma * self.config.gae_lambda * continuation * gae
                trajectory_advantages[index] = gae
                next_value = value
            entries.extend(trajectory)
            advantages_parts.append(trajectory_advantages)
            returns_parts.append(trajectory_advantages + np.asarray(
                [entry[4] for entry in trajectory], dtype=np.float32))
        advantages = np.concatenate(advantages_parts)
        returns = np.concatenate(returns_parts)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        states = torch.as_tensor(np.stack([entry[0] for entry in entries]), device=self.device)
        actions = torch.as_tensor([entry[1] for entry in entries], device=self.device, dtype=torch.long)
        old_logs = torch.as_tensor([entry[3] for entry in entries], device=self.device)
        advantage_tensor = torch.as_tensor(advantages, device=self.device)
        return_tensor = torch.as_tensor(returns, device=self.device)
        indices = np.arange(len(entries))
        losses = []
        for _ in range(self.config.ppo_epochs):
            self.random.shuffle(indices)
            for start in range(0, len(indices), self.config.ppo_minibatch_size):
                selected = torch.as_tensor(indices[start:start + self.config.ppo_minibatch_size],
                                            device=self.device, dtype=torch.long)
                logits, values = self.policy(states[selected])
                distribution = Categorical(logits=logits)
                new_logs = distribution.log_prob(actions[selected])
                ratio = (new_logs - old_logs[selected]).exp()
                unclipped = ratio * advantage_tensor[selected]
                clipped = ratio.clamp(1 - self.config.ppo_clip, 1 + self.config.ppo_clip) * advantage_tensor[selected]
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                critic_loss = nn.functional.mse_loss(values, return_tensor[selected])
                entropy_bonus = distribution.entropy().mean()
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                losses.append(float(loss.detach().cpu()))
        self.rollout.clear()
        self.updates += 1
        return float(np.mean(losses)) if losses else None

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        _atomic_torch_save({"algorithm": self.algorithm, "config": asdict(self.config),
                            "policy": self.policy.state_dict(), "optimizer": self.optimizer.state_dict(),
                            "steps": self.steps, "updates": self.updates,
                            "numpy_random": self.random.bit_generator.state,
                            "torch_random": torch.get_rng_state()}, destination)

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("algorithm") != self.algorithm or payload.get("config") != asdict(self.config):
            raise ValueError("PPO checkpoint algorithm/configuration mismatch")
        self.policy.load_state_dict(payload["policy"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.steps, self.updates = int(payload["steps"]), int(payload["updates"])
        # The checkpoint has no emulator savestates, so an in-flight rollout
        # would no longer have valid next states after restart.
        self.rollout = {}
        if "numpy_random" in payload:
            self.random.bit_generator.state = payload["numpy_random"]
        if isinstance(payload.get("torch_random"), torch.Tensor):
            torch.set_rng_state(payload["torch_random"])


def load_baseline_policy(path: str | Path, device: str | None = None):
    """Load a trained local DDQN or PPO baseline as a deterministic policy."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    algorithm = payload.get("algorithm", "")
    configuration = BaselineConfig(**payload["config"])
    target_device = torch.device(device or ("mps" if torch.backends.mps.is_available()
                                            else "cuda" if torch.cuda.is_available() else "cpu"))
    if algorithm.startswith("Basic DDQN"):
        policy = BasicQNetwork(configuration.observation_size, configuration.hidden_size,
                              configuration.action_count).to(target_device)
        policy.load_state_dict(payload["online"])

        class LoadedQPolicy:
            def select_actions(self, states: np.ndarray, explore: bool = False) -> np.ndarray:
                observations = torch.as_tensor(np.asarray(states, dtype=np.float32).reshape(
                    -1, configuration.observation_size), device=target_device)
                with torch.no_grad():
                    return policy(observations).argmax(dim=1).cpu().numpy().astype(np.int64)

        policy.eval()
        return algorithm, LoadedQPolicy()
    if algorithm.startswith("PPO"):
        policy = ActorCriticNetwork(configuration.observation_size, configuration.hidden_size,
                                    configuration.action_count).to(target_device)
        policy.load_state_dict(payload["policy"])

        class LoadedPPOPolicy:
            def select_actions(self, states: np.ndarray, explore: bool = False) -> np.ndarray:
                observations = torch.as_tensor(np.asarray(states, dtype=np.float32).reshape(
                    -1, configuration.observation_size), device=target_device)
                with torch.no_grad():
                    logits, _ = policy(observations)
                return logits.argmax(dim=1).cpu().numpy().astype(np.int64)

        policy.eval()
        return algorithm, LoadedPPOPolicy()
    raise ValueError(f"unsupported baseline checkpoint algorithm: {algorithm!r}")
