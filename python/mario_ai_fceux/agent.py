"""Complete Rainbow DQN: C51, Double DQN, dueling, NoisyNets, PER and n-step."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import os
import random
import shutil
from pathlib import Path
from typing import Deque

import numpy as np
import torch
from torch import nn

from .actions import (ACTION_COUNT, ACTION_DURATIONS,
                      LEGACY_ACTION_COUNT, greedy_action)
from .model import RainbowNetwork
from .replay import PrioritizedReplayBuffer, Transition


@dataclass
class AgentConfig:
    observation_size: int = 184
    action_count: int = ACTION_COUNT
    gamma: float = 0.99
    # The Ape-X reference uses 6.25e-4 with a 512 batch; the bootcamp learner
    # started ten times below that and, at ~50 updates/s, was slow to move away
    # from the policy the old shaping had already fitted.
    learning_rate: float = 1.25e-4
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
        self.action_space_migrated = False

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
            values = self.online(torch.from_numpy(states).to(self.device), self.support).cpu().numpy()
        if self.config.action_count == ACTION_COUNT:
            return np.asarray([greedy_action(row) for row in values], dtype=np.int64)
        return values.argmax(axis=1).astype(np.int64)

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
                discount *= transition.discount
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
        replay_snapshot_id = None
        try:
            if replay_path is not None:
                replay_path = Path(replay_path)
                replay_path.parent.mkdir(parents=True, exist_ok=True)
                replay_backup = replay_path.with_suffix(replay_path.suffix + ".bak")
                checkpoint_backup = path.with_suffix(path.suffix + ".bak")
                # Preserve the previous pair before replacing either member.
                if replay_path.exists() and path.exists():
                    replay_copy = replay_backup.with_suffix(replay_backup.suffix + ".tmp")
                    checkpoint_copy = checkpoint_backup.with_suffix(checkpoint_backup.suffix + ".tmp")
                    shutil.copy2(replay_path, replay_copy)
                    shutil.copy2(path, checkpoint_copy)
                    os.replace(replay_copy, replay_backup)
                    os.replace(checkpoint_copy, checkpoint_backup)
                replay_snapshot_id = self.replay.save(replay_path)
            torch.save({"algorithm": "Rainbow DQN (C51 + NoisyNet + Double + Dueling + PER + n-step)",
                        "config": asdict(self.config), "steps": self.steps, "optimizer_steps": self.optimizer_steps,
                        "online": self.online.state_dict(), "target": self.target.state_dict(),
                        "optimizer": self.optimizer.state_dict(), "python_random": random.getstate(),
                        "numpy_random": np.random.get_state(), "torch_random": torch.get_rng_state(),
                        "torch_cuda_random": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        "torch_mps_random": (torch.mps.get_rng_state()
                                             if torch.backends.mps.is_available()
                                             and hasattr(torch.mps, "get_rng_state") else None),
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "replay_snapshot_id": replay_snapshot_id}, temporary)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def load_checkpoint_pair(checkpoint_path: str | Path, replay_path: str | Path,
                             observation_size: int, seed: int = 0,
                             capacity: int = 100_000) -> tuple[Path, PrioritizedReplayBuffer]:
        """Load a matching model/replay pair, falling back to the prior pair."""
        checkpoint_path, replay_path = Path(checkpoint_path), Path(replay_path)
        candidates = (
            (checkpoint_path, replay_path),
            (checkpoint_path.with_suffix(checkpoint_path.suffix + ".bak"),
             replay_path.with_suffix(replay_path.suffix + ".bak")),
        )
        failures = []
        for model_candidate, replay_candidate in candidates:
            if not model_candidate.exists() or not replay_candidate.exists():
                continue
            try:
                payload = torch.load(model_candidate, map_location="cpu", weights_only=False)
                replay = PrioritizedReplayBuffer.load(replay_candidate, seed)
                expected_snapshot = payload.get("replay_snapshot_id")
                if expected_snapshot is not None and replay.snapshot_id != expected_snapshot:
                    raise ValueError("model and replay snapshot IDs differ")
                return model_candidate, replay
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                failures.append(f"{model_candidate.name}: {exc}")
        if failures:
            raise RuntimeError("no consistent Rainbow model/replay checkpoint pair: " + "; ".join(failures))
        if replay_path.exists():
            return checkpoint_path, PrioritizedReplayBuffer.load(replay_path, seed)
        return checkpoint_path, PrioritizedReplayBuffer(observation_size, capacity=capacity, seed=seed)

    @staticmethod
    def _expand_legacy_action_head(old_state: dict[str, torch.Tensor],
                                   new_state: dict[str, torch.Tensor],
                                   atom_count: int) -> dict[str, torch.Tensor]:
        """Copy six trained SMB1 action heads into the 12-frame choices."""
        migrated = {key: value.clone() for key, value in new_state.items()}
        for key, old_value in old_state.items():
            if key not in migrated:
                continue
            new_value = migrated[key]
            if old_value.shape == new_value.shape:
                migrated[key] = old_value
                continue
            if not key.startswith("advantage_output.") or old_value.ndim < 1:
                raise ValueError(f"unsupported legacy checkpoint tensor shape: {key}")
            if old_value.shape[0] != LEGACY_ACTION_COUNT * atom_count:
                raise ValueError(f"unexpected legacy action-head shape for {key}")
            if new_value.shape[0] != ACTION_COUNT * atom_count:
                raise ValueError(f"unexpected new action-head shape for {key}")
            for new_action in range(ACTION_COUNT):
                base_index = new_action // len(ACTION_DURATIONS)
                old_action = base_index if base_index < LEGACY_ACTION_COUNT else 1
                source_start = old_action * atom_count
                target_start = new_action * atom_count
                migrated[key][target_start:target_start + atom_count] = \
                    old_value[source_start:source_start + atom_count]
        return migrated

    def _migrate_legacy_replay(self) -> None:
        """Map old fixed-duration actions to their equivalent 12-frame IDs."""
        old_actions = self.replay.actions[:self.replay.size].astype(np.int64)
        if np.any(old_actions < 0) or np.any(old_actions >= LEGACY_ACTION_COUNT):
            raise ValueError("legacy replay contains an action outside the old 0..5 range")
        normal_duration = ACTION_DURATIONS.index(12)
        self.replay.actions[:self.replay.size] = (
            old_actions * len(ACTION_DURATIONS) + normal_duration
        ).astype(self.replay.actions.dtype)

    def load(self, path: str | Path, *, validate_replay: bool = True,
             restore_rng: bool = True) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        saved_config = payload.get("config")
        legacy_action_migration = False
        if saved_config is not None:
            normalized_saved_config = asdict(AgentConfig(**saved_config))
            current_config = asdict(self.config)
            differences = [name for name, value in current_config.items()
                           if normalized_saved_config.get(name) != value]
            if differences == ["action_count"] and normalized_saved_config["action_count"] == LEGACY_ACTION_COUNT \
                    and self.config.action_count == ACTION_COUNT:
                legacy_action_migration = True
            elif differences:
                raise ValueError("checkpoint AgentConfig differs in: " + ", ".join(differences))
        snapshot_id = payload.get("replay_snapshot_id")
        if validate_replay and snapshot_id is not None and self.replay.snapshot_id != snapshot_id:
            raise ValueError("checkpoint model and replay snapshot do not match")
        if legacy_action_migration:
            self.online.load_state_dict(self._expand_legacy_action_head(
                payload["online"], self.online.state_dict(), self.config.atom_count
            ))
            self.target.load_state_dict(self._expand_legacy_action_head(
                payload["target"], self.target.state_dict(), self.config.atom_count
            ))
            self._migrate_legacy_replay()
            self.action_space_migrated = True
            # Adam moments have the old output-head shape; reset optimizer
            # moments while retaining network weights, replay, and step count.
        else:
            self.online.load_state_dict(payload["online"])
            self.target.load_state_dict(payload["target"])
            self.optimizer.load_state_dict(payload["optimizer"])
        self.steps, self.optimizer_steps = int(payload.get("steps", 0)), int(payload.get("optimizer_steps", 0))
        if restore_rng and "python_random" in payload:
            random.setstate(payload["python_random"])
            np.random.set_state(payload["numpy_random"])
            torch_state = payload["torch_random"]
            # Checkpoints created before the reproducibility format may carry
            # a non-Tensor value here. Keep their trained weights, but do not
            # crash a resume over an unusable old RNG record.
            if isinstance(torch_state, torch.Tensor) and torch_state.dtype == torch.uint8:
                torch.set_rng_state(torch_state.cpu())
            cuda_state = payload.get("torch_cuda_random")
            if torch.cuda.is_available() and cuda_state:
                torch.cuda.set_rng_state_all(cuda_state)
            mps_state = payload.get("torch_mps_random")
            if (torch.backends.mps.is_available() and mps_state is not None
                    and hasattr(torch.mps, "set_rng_state")):
                torch.mps.set_rng_state(mps_state.cpu())
            if "deterministic_algorithms" in payload:
                torch.use_deterministic_algorithms(bool(payload["deterministic_algorithms"]))


# Old import name remains available for external users; it now implements full Rainbow.
RainbowLiteAgent = RainbowAgent
