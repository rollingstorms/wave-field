from __future__ import annotations

import torch
from torch import nn

from .encoding import ACTION_SIZE, BOARD_CHANNELS, SIDE_SIZE


BOARD_TOKEN_COUNT = 7 * 7


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
        self.transformer = None
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
        elif architecture == "transformer":
            board_features = hidden_size
            heads = 4 if hidden_size % 4 == 0 else 1
            self.square_projection = nn.Linear(board_channels, hidden_size)
            self.side_token = nn.Linear(side_size, hidden_size)
            self.position_embedding = nn.Parameter(torch.zeros(1, BOARD_TOKEN_COUNT + 1, hidden_size))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=heads,
                dim_feedforward=hidden_size * 4,
                dropout=0.0,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.board = nn.Flatten()
        else:
            raise ValueError(f"Unknown model architecture '{architecture}'")
        if architecture != "transformer":
            self.side = nn.Sequential(
                nn.Linear(side_size, 32),
                nn.ReLU(),
            )
        self.trunk = nn.Sequential(
            nn.Linear(
                board_features * (BOARD_TOKEN_COUNT + 1)
                if architecture == "transformer"
                else board_features * 7 * 7 + 32,
                hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden_size, ACTION_SIZE)
        self.value = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())

    def forward(self, board: torch.Tensor, side: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.architecture == "transformer":
            squares = board.permute(0, 2, 3, 1).reshape(board.shape[0], BOARD_TOKEN_COUNT, self.board_channels)
            square_tokens = self.square_projection(squares)
            side_token = self.side_token(side).unsqueeze(1)
            tokens = torch.cat([side_token, square_tokens], dim=1) + self.position_embedding
            assert self.transformer is not None
            embedding = self.transformer(tokens).flatten(start_dim=1)
        else:
            embedding = torch.cat([self.board(board), self.side(side)], dim=1)
        hidden = self.trunk(embedding)
        return self.policy(hidden), self.value(hidden).squeeze(-1)


def masked_policy_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(legal_mask <= 0, -1.0e9)
