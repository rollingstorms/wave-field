from __future__ import annotations

import torch
from torch import nn

from .encoding import ACTION_SIZE, BOARD_CHANNELS, SIDE_SIZE, TUNING_ACTION_SIZE


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
        history_plies: int = 1,
    ) -> None:
        super().__init__()
        self.board_channels = board_channels
        self.side_size = side_size
        self.architecture = architecture
        self.history_plies = max(1, int(history_plies))
        self.transformer = None
        self.temporal_transformer = None
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
        elif architecture == "sequence_transformer":
            board_features = hidden_size
            heads = 4 if hidden_size % 4 == 0 else 1
            self.square_projection = nn.Linear(board_channels, hidden_size)
            self.side_token = nn.Linear(side_size, hidden_size)
            self.position_embedding = nn.Parameter(torch.zeros(1, BOARD_TOKEN_COUNT + 1, hidden_size))
            state_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=heads,
                dim_feedforward=hidden_size * 4,
                dropout=0.0,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(state_layer, num_layers=2)
            self.temporal_position_embedding = nn.Parameter(torch.zeros(1, self.history_plies, hidden_size))
            temporal_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=heads,
                dim_feedforward=hidden_size * 4,
                dropout=0.0,
                batch_first=True,
                activation="gelu",
            )
            self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=2)
            self.board = nn.Flatten()
        else:
            raise ValueError(f"Unknown model architecture '{architecture}'")
        if architecture not in ("transformer", "sequence_transformer"):
            self.side = nn.Sequential(
                nn.Linear(side_size, 32),
                nn.ReLU(),
            )
        self.trunk = nn.Sequential(
            nn.Linear(
                board_features * (BOARD_TOKEN_COUNT + 1)
                if architecture == "transformer"
                else board_features
                if architecture == "sequence_transformer"
                else board_features * 7 * 7 + 32,
                hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden_size, ACTION_SIZE)
        self.action_kind = nn.Linear(hidden_size, 2)
        self.tuning_policy = nn.Linear(hidden_size, TUNING_ACTION_SIZE)
        self.value = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())

    def _encode_board_tokens(self, board: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        squares = board.permute(0, 2, 3, 1).reshape(board.shape[0], BOARD_TOKEN_COUNT, self.board_channels)
        square_tokens = self.square_projection(squares)
        side_token = self.side_token(side).unsqueeze(1)
        tokens = torch.cat([side_token, square_tokens], dim=1) + self.position_embedding
        assert self.transformer is not None
        return self.transformer(tokens)

    def _sequence_inputs(
        self,
        board: torch.Tensor,
        side: torch.Tensor,
        history_board: torch.Tensor | None,
        history_side: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if history_board is None or history_side is None:
            if board.ndim == 5 and side.ndim == 3:
                history_board = board
                history_side = side
            else:
                history_board = board.unsqueeze(1)
                history_side = side.unsqueeze(1)
        if history_board.shape[1] > self.history_plies:
            history_board = history_board[:, -self.history_plies:]
            history_side = history_side[:, -self.history_plies:]
        if history_board.shape[1] < self.history_plies:
            pad = self.history_plies - history_board.shape[1]
            board_pad = history_board[:, :1].new_zeros(
                history_board.shape[0],
                pad,
                history_board.shape[2],
                history_board.shape[3],
                history_board.shape[4],
            )
            side_pad = history_side[:, :1].new_zeros(history_side.shape[0], pad, history_side.shape[2])
            history_board = torch.cat([board_pad, history_board], dim=1)
            history_side = torch.cat([side_pad, history_side], dim=1)
        return history_board, history_side

    def encode_hidden(
        self,
        board: torch.Tensor,
        side: torch.Tensor,
        history_board: torch.Tensor | None = None,
        history_side: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.architecture == "transformer":
            embedding = self._encode_board_tokens(board, side).flatten(start_dim=1)
        elif self.architecture == "sequence_transformer":
            history_board, history_side = self._sequence_inputs(board, side, history_board, history_side)
            batch, steps = history_board.shape[:2]
            flat_board = history_board.reshape(batch * steps, *history_board.shape[2:])
            flat_side = history_side.reshape(batch * steps, history_side.shape[-1])
            state_tokens = self._encode_board_tokens(flat_board, flat_side)
            state_embeddings = state_tokens[:, 0, :].reshape(batch, steps, -1)
            temporal_tokens = state_embeddings + self.temporal_position_embedding[:, -steps:, :]
            assert self.temporal_transformer is not None
            embedding = self.temporal_transformer(temporal_tokens)[:, -1, :]
        else:
            embedding = torch.cat([self.board(board), self.side(side)], dim=1)
        return self.trunk(embedding)

    def forward(
        self,
        board: torch.Tensor,
        side: torch.Tensor,
        history_board: torch.Tensor | None = None,
        history_side: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode_hidden(board, side, history_board=history_board, history_side=history_side)
        return self.policy(hidden), self.value(hidden).squeeze(-1)

    def full_policy(
        self,
        board: torch.Tensor,
        side: torch.Tensor,
        history_board: torch.Tensor | None = None,
        history_side: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encode_hidden(board, side, history_board=history_board, history_side=history_side)
        return self.action_kind(hidden), self.policy(hidden), self.tuning_policy(hidden)


def masked_policy_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(legal_mask <= 0, -1.0e9)
