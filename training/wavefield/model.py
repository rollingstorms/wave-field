from __future__ import annotations

import torch
from torch import nn

from .encoding import ACTION_SIZE, BOARD_CHANNELS, SIDE_SIZE


class PolicyValueNet(nn.Module):
    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        self.board = nn.Sequential(
            nn.Conv2d(BOARD_CHANNELS, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.side = nn.Sequential(
            nn.Linear(SIDE_SIZE, 32),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(64 * 7 * 7 + 32, hidden_size),
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
