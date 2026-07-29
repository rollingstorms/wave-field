from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from .engine import Action, RustEngine


BOARD_SIZE = 7
PIECE_TYPES = ("pawn", "rook", "spy", "king")
PLAYERS = ("red", "blue")
PIECE_IDS = (
    "blue-rook-1",
    "blue-king-1",
    "blue-rook-2",
    "blue-pawn-1",
    "blue-spy-1",
    "blue-pawn-2",
    "red-pawn-1",
    "red-spy-1",
    "red-pawn-2",
    "red-rook-1",
    "red-king-1",
    "red-rook-2",
)

OCCUPANCY_CHANNELS = len(PLAYERS) * len(PIECE_TYPES)
RED_UNSTABLE_CHANNEL = OCCUPANCY_CHANNELS
BLUE_UNSTABLE_CHANNEL = OCCUPANCY_CHANNELS + 1
FIELD_SIGNED_CHANNEL = OCCUPANCY_CHANNELS + 2
FIELD_MAGNITUDE_CHANNEL = OCCUPANCY_CHANNELS + 3
CURRENT_PLAYER_CHANNEL = OCCUPANCY_CHANNELS + 4
BOARD_CHANNELS = OCCUPANCY_CHANNELS + 5

TUNING_SIZE_PER_PLAYER = sum({"pawn": 1, "rook": 2, "spy": 3, "king": 3}.values())
SIDE_SIZE = TUNING_SIZE_PER_PLAYER * len(PLAYERS) + 1
ACTION_SIZE = len(PIECE_IDS) * BOARD_SIZE * BOARD_SIZE


def piece_slot(piece_id: str) -> int:
    return PIECE_IDS.index(piece_id)


def square_index(position: Dict[str, int]) -> int:
    return position["y"] * BOARD_SIZE + position["x"]


def action_index(action: Action) -> int:
    return piece_slot(action["pieceId"]) * BOARD_SIZE * BOARD_SIZE + square_index(action["destination"])


def decode_action(index: int) -> Action:
    slot, destination_index = divmod(index, BOARD_SIZE * BOARD_SIZE)
    y, x = divmod(destination_index, BOARD_SIZE)
    return {"pieceId": PIECE_IDS[slot], "destination": {"x": x, "y": y}}


def _occupancy_channel(owner: str, piece_type: str) -> int:
    return PLAYERS.index(owner) * len(PIECE_TYPES) + PIECE_TYPES.index(piece_type)


def _flat_components(state: Dict[str, Any]) -> List[float]:
    values: List[float] = []
    for player in PLAYERS:
        for piece_type in PIECE_TYPES:
            values.extend(float(value) for value in state["components"][player][piece_type])
    values.append(1.0 if state["currentPlayer"] == "red" else -1.0)
    return values


def legal_action_mask(actions: Iterable[Action]) -> np.ndarray:
    mask = np.zeros((ACTION_SIZE,), dtype=np.float32)
    for action in actions:
        mask[action_index(action)] = 1.0
    return mask


def encode_state(
    state: Dict[str, Any],
    engine: RustEngine,
    legal_actions: Iterable[Action] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    board = np.zeros((BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for piece in state["pieces"]:
        x = piece["position"]["x"]
        y = piece["position"]["y"]
        board[_occupancy_channel(piece["owner"], piece["type"]), y, x] = 1.0
        if piece.get("unstable", False):
            unstable_channel = RED_UNSTABLE_CHANNEL if piece["owner"] == "red" else BLUE_UNSTABLE_CHANNEL
            board[unstable_channel, y, x] = 1.0

    field = np.asarray(engine.evaluate_field(state), dtype=np.float32)
    board[FIELD_SIGNED_CHANNEL] = np.clip(field / 8.0, -1.0, 1.0)
    board[FIELD_MAGNITUDE_CHANNEL] = np.clip(np.abs(field) / 8.0, 0.0, 1.0)
    board[CURRENT_PLAYER_CHANNEL].fill(1.0 if state["currentPlayer"] == "red" else -1.0)

    side = np.asarray(_flat_components(state), dtype=np.float32)
    actions = list(legal_actions) if legal_actions is not None else engine.legal_actions(state)
    return board, side, legal_action_mask(actions)
