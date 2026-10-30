"""Square-board symmetries and color exchange for sparse-rule training."""
from dataclasses import replace

import numpy as np

from .encoding import (BOARD_CHANNELS, RICH_BOARD_CHANNELS,
                       FIELD_SIGNED_CHANNEL, CURRENT_PLAYER_CHANNEL)
from .selfplay import Sample

# Counterparts at rotated starting squares: the numbered pawns/rooks exchange 1/2.
PIECE_PERMUTATION = np.asarray([11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
MOVE_PERMUTATION = (PIECE_PERMUTATION[:, None] * 49 + np.arange(48, -1, -1)).reshape(-1)


def swap_board(board: np.ndarray) -> np.ndarray:
    channels = board.shape[-3]
    if channels not in (BOARD_CHANNELS, RICH_BOARD_CHANNELS):
        raise ValueError(f'Unsupported board channel count: {channels}')
    order = list(range(4, 8)) + list(range(4)) + [9, 8, 10, 11, 12]
    if channels == RICH_BOARD_CHANNELS:
        order += (BOARD_CHANNELS + PIECE_PERMUTATION).tolist()
    result = np.take(board[..., ::-1, ::-1], order, axis=-3).copy()
    result[..., FIELD_SIGNED_CHANNEL, :, :] *= -1
    result[..., CURRENT_PLAYER_CHANNEL, :, :] *= -1
    return result


def swap_side(side: np.ndarray) -> np.ndarray:
    result = np.concatenate((side[..., 7:14], side[..., :7], -side[..., 14:15]), axis=-1)
    return result.copy()


def swap_sample(sample: Sample) -> Sample:
    """Value and tuning targets are mover-relative and remain unchanged."""
    return replace(sample,
        board=swap_board(sample.board), side=swap_side(sample.side),
        player='red' if sample.player == 'blue' else 'blue',
        legal_mask=sample.legal_mask[MOVE_PERMUTATION].copy(),
        action_index=int(MOVE_PERMUTATION[sample.action_index]) if sample.action_index >= 0 else sample.action_index,
        move_policy=sample.move_policy[MOVE_PERMUTATION].copy() if sample.move_policy is not None else None,
        history_board=swap_board(sample.history_board) if sample.history_board is not None else None,
        history_side=swap_side(sample.history_side) if sample.history_side is not None else None,
        metadata={**sample.metadata, 'color_augmentation': 'rotated_color_swap'},
    )


def transform_grid(grid: np.ndarray, symmetry: int) -> np.ndarray:
    """D4: four counterclockwise rotations, optionally after a left/right reflection."""
    if not 0 <= symmetry < 8:
        raise ValueError('Symmetry must be in [0, 8)')
    reflected = grid[..., :, ::-1] if symmetry >= 4 else grid
    return np.rot90(reflected, symmetry % 4, axes=(-2, -1)).copy()


def spatial_sample(sample: Sample, symmetry: int) -> Sample:
    pull = transform_grid(np.arange(49).reshape(7, 7), symmetry).reshape(-1)
    push = np.argsort(pull)
    def moves(array):
        return array.reshape(-1, 49)[:, pull].reshape(array.shape).copy()
    action = sample.action_index
    if action >= 0:
        slot, square = divmod(action, 49)
        action = int(slot * 49 + push[square])
    return replace(sample,
        board=transform_grid(sample.board, symmetry),
        legal_mask=moves(sample.legal_mask), action_index=action,
        move_policy=moves(sample.move_policy) if sample.move_policy is not None else None,
        history_board=transform_grid(sample.history_board, symmetry) if sample.history_board is not None else None,
        metadata={**sample.metadata, 'spatial_symmetry': symmetry})


def paired_samples(samples: list[Sample], seed: int, spatial: bool = False) -> list[Sample]:
    """Use half the draw as originals plus mirrors, keeping the update row budget."""
    if not samples:
        return []
    count = max(1, len(samples) // 2)
    rng = np.random.default_rng(seed)
    indexes = rng.choice(len(samples), count, replace=False)
    # Balanced orientations across a batch, freshly assigned each iteration.
    orientations = rng.permutation(np.arange(count) % 8) if spatial else np.zeros(count, dtype=int)
    result = []
    for index, symmetry in zip(indexes, orientations):
        original = samples[int(index)]
        pair = (original, swap_sample(original))
        result.extend(spatial_sample(row, int(symmetry)) if spatial else row for row in pair)
    return result
