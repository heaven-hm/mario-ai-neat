"""Compact dueling Q-network for SMB1's 184 RAM/tile observations."""

from __future__ import annotations

import torch
from torch import nn


class DuelingQNetwork(nn.Module):
    def __init__(self, observation_size: int = 184, action_count: int = 6) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observation_size, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_count))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(observations)
        advantage = self.advantage(encoded)
        return self.value(encoded) + advantage - advantage.mean(dim=1, keepdim=True)
