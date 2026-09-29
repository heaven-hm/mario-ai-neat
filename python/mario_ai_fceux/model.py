"""Noisy, dueling distributional network used by the full Rainbow agent."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as functional


class NoisyLinear(nn.Module):
    """Factorised Gaussian exploration layer from the NoisyNet paper."""

    def __init__(self, input_features: int, output_features: int, sigma_zero: float = 0.5) -> None:
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight_mu = nn.Parameter(torch.empty(output_features, input_features))
        self.weight_sigma = nn.Parameter(torch.empty(output_features, input_features))
        self.bias_mu = nn.Parameter(torch.empty(output_features))
        self.bias_sigma = nn.Parameter(torch.empty(output_features))
        self.register_buffer("weight_epsilon", torch.empty(output_features, input_features))
        self.register_buffer("bias_epsilon", torch.empty(output_features))
        bound = 1 / math.sqrt(input_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(sigma_zero / math.sqrt(input_features))
        self.bias_sigma.data.fill_(sigma_zero / math.sqrt(output_features))
        self.reset_noise()

    @staticmethod
    def _scaled_noise(size: int, device: torch.device) -> torch.Tensor:
        noise = torch.randn(size, device=device)
        return noise.sign() * noise.abs().sqrt()

    def reset_noise(self) -> None:
        input_noise = self._scaled_noise(self.input_features, self.weight_epsilon.device)
        output_noise = self._scaled_noise(self.output_features, self.weight_epsilon.device)
        self.weight_epsilon.copy_(torch.outer(output_noise, input_noise))
        self.bias_epsilon.copy_(output_noise)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.training:
            return functional.linear(value, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                                     self.bias_mu + self.bias_sigma * self.bias_epsilon)
        return functional.linear(value, self.weight_mu, self.bias_mu)


class RainbowNetwork(nn.Module):
    """Dueling C51 categorical value network with NoisyNet exploration."""

    def __init__(self, observation_size: int = 184, action_count: int = 6, atom_count: int = 51) -> None:
        super().__init__()
        self.action_count = action_count
        self.atom_count = atom_count
        self.encoder = nn.Sequential(nn.Linear(observation_size, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.value_hidden = NoisyLinear(256, 128)
        self.value_output = NoisyLinear(128, atom_count)
        self.advantage_hidden = NoisyLinear(256, 128)
        self.advantage_output = NoisyLinear(128, action_count * atom_count)

    def logits(self, observations: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(observations)
        value = self.value_output(functional.relu(self.value_hidden(encoded))).view(-1, 1, self.atom_count)
        advantage = self.advantage_output(functional.relu(self.advantage_hidden(encoded))).view(
            -1, self.action_count, self.atom_count)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def distribution(self, observations: torch.Tensor) -> torch.Tensor:
        probabilities = functional.softmax(self.logits(observations), dim=-1).clamp_min(1e-6)
        return probabilities / probabilities.sum(dim=-1, keepdim=True)

    def forward(self, observations: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        return (self.distribution(observations) * support.view(1, 1, -1)).sum(dim=-1)

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# Existing imports keep working; its outputs are now categorical Rainbow Q-values.
DuelingQNetwork = RainbowNetwork
