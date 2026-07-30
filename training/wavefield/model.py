from __future__ import annotations

import torch
from torch import nn

from .encoding import ACTION_SIZE, BOARD_CHANNELS, SIDE_SIZE


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.activation = nn.ReLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.layers(value))


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        board_channels: int = BOARD_CHANNELS,
        side_size: int = SIDE_SIZE,
        architecture: str = "conv",
    ) -> None:
        super().__init__()
        self.board_channels = board_channels
        self.side_size = side_size
        self.architecture = architecture
        if architecture == "conv":
            board_features = 64
            self.board = nn.Sequential(
                nn.Conv2d(board_channels, 48, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(48, board_features, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
        elif architecture == "residual":
            board_features = 64
            self.board = nn.Sequential(
                nn.Conv2d(board_channels, board_features, kernel_size=3, padding=1),
                nn.ReLU(),
                ResidualBlock(board_features),
                ResidualBlock(board_features),
                nn.Flatten(),
            )
        else:
            raise ValueError(f"Unknown model architecture '{architecture}'")
        self.side = nn.Sequential(
            nn.Linear(side_size, 32),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(board_features * 7 * 7 + 32, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden_size, ACTION_SIZE)
        self.value = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())

    def forward(self, board: torch.Tensor, side: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = torch.cat([self.board(board), self.side(side)], dim=1)
        hidden = self.trunk(embedding)
        return self.policy(hidden), self.value(hidden).squeeze(-1)


def masked_policy_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(legal_mask <= 0, -1.0e9)
