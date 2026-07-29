from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from .encoding import action_index, encode_state
from .engine import Action, RustEngine, load_initial_state


@dataclass
class Sample:
    board: np.ndarray
    side: np.ndarray
    legal_mask: np.ndarray
    action_index: int
    player: str
    value: float = 0.0


def result_value(status: str, player: str) -> float:
    if status == "playing":
        return 0.0
    winner = "red" if status == "red-won" else "blue"
    return 1.0 if winner == player else -1.0


def random_selfplay_game(
    engine: RustEngine,
    max_plies: int = 160,
    seed: int | None = None,
    initial_state: Dict[str, Any] | None = None,
) -> List[Sample]:
    rng = random.Random(seed)
    state = initial_state or load_initial_state()
    samples: List[Sample] = []

    for _ply in range(max_plies):
        if state["status"] != "playing":
            break
        actions = engine.legal_actions(state)
        if not actions:
            break
        action: Action = rng.choice(actions)
        board, side, legal_mask = encode_state(state, engine, actions)
        samples.append(
            Sample(
                board=board,
                side=side,
                legal_mask=legal_mask,
                action_index=action_index(action),
                player=state["currentPlayer"],
            )
        )
        state = engine.apply_action(state, action, analyze_checkmate=False)

    for sample in samples:
        sample.value = result_value(state["status"], sample.player)
    return samples


def random_selfplay_samples(
    engine: RustEngine,
    games: int,
    max_plies: int = 160,
    seed: int = 0,
) -> List[Sample]:
    samples: List[Sample] = []
    for game in range(games):
        samples.extend(random_selfplay_game(engine, max_plies=max_plies, seed=seed + game))
    return samples
