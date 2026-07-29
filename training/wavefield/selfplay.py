from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch

from .encoding import action_index, decode_action, encode_state, legal_action_mask
from .engine import Action, RustEngine, load_initial_state
from .model import PolicyValueNet, masked_policy_logits


Player = Literal["red", "blue"]
PolicyMode = Literal["random", "model", "heuristic"]
CapValueMode = Literal["zero", "material"]


@dataclass
class Sample:
    board: np.ndarray
    side: np.ndarray
    legal_mask: np.ndarray
    action_index: int
    player: str
    value: float = 0.0


@dataclass
class GameStats:
    plies: int = 0
    status: str = "playing"
    winner: Optional[str] = None
    first_loss_player: Optional[str] = None
    first_loss_piece_type: Optional[str] = None
    losses_by_player: Dict[str, int] = field(default_factory=lambda: {"red": 0, "blue": 0})
    losses_by_piece_type: Dict[str, int] = field(default_factory=dict)
    rescue_opportunities: int = 0
    rescues: int = 0
    pressure_sum: Dict[str, int] = field(default_factory=lambda: {"red": 0, "blue": 0})
    pressure_samples: int = 0
    min_winner_pieces: Optional[int] = None
    max_loser_pieces: Optional[int] = None
    final_piece_counts: Dict[str, int] = field(default_factory=lambda: {"red": 0, "blue": 0})

    @property
    def decisive(self) -> bool:
        return self.winner is not None

    @property
    def rescue_rate(self) -> float:
        if self.rescue_opportunities == 0:
            return 0.0
        return self.rescues / self.rescue_opportunities

    def to_dict(self) -> Dict[str, Any]:
        avg_pressure = {
            player: (total / self.pressure_samples if self.pressure_samples else 0.0)
            for player, total in self.pressure_sum.items()
        }
        return {
            "plies": self.plies,
            "status": self.status,
            "winner": self.winner,
            "decisive": self.decisive,
            "first_loss_player": self.first_loss_player,
            "first_loss_piece_type": self.first_loss_piece_type,
            "losses_by_player": self.losses_by_player,
            "losses_by_piece_type": self.losses_by_piece_type,
            "rescue_opportunities": self.rescue_opportunities,
            "rescues": self.rescues,
            "rescue_rate": self.rescue_rate,
            "avg_pressure": avg_pressure,
            "min_winner_pieces": self.min_winner_pieces,
            "max_loser_pieces": self.max_loser_pieces,
            "final_piece_counts": self.final_piece_counts,
        }


@dataclass
class GameRecord:
    samples: List[Sample]
    stats: GameStats
    final_state: Dict[str, Any]


def result_value(status: str, player: str) -> float:
    if status == "playing":
        return 0.0
    winner = "red" if status == "red-won" else "blue"
    return 1.0 if winner == player else -1.0


def _winner(status: str) -> Optional[str]:
    if status == "red-won":
        return "red"
    if status == "blue-won":
        return "blue"
    return None


def _piece_map(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {piece["id"]: piece for piece in state["pieces"]}


def _piece_counts(state: Dict[str, Any]) -> Dict[str, int]:
    counts = {"red": 0, "blue": 0}
    for piece in state["pieces"]:
        counts[piece["owner"]] += 1
    return counts


def _unstable_ids_for_player(state: Dict[str, Any], player: str) -> set[str]:
    return {
        piece["id"]
        for piece in state["pieces"]
        if piece["owner"] == player and piece.get("unstable", False)
    }


def _material_value(state: Dict[str, Any], player: str) -> float:
    counts = _piece_counts(state)
    opponent = "blue" if player == "red" else "red"
    return max(-1.0, min(1.0, (counts[player] - counts[opponent]) / 6.0))


def _assign_values(samples: List[Sample], state: Dict[str, Any], cap_value: CapValueMode) -> None:
    for sample in samples:
        value = result_value(state["status"], sample.player)
        if state["status"] == "playing" and cap_value == "material":
            value = _material_value(state, sample.player)
        sample.value = value


def _sample_from_action(
    state: Dict[str, Any],
    engine: RustEngine,
    actions: List[Action],
    action: Action,
) -> Sample:
    board, side, mask = encode_state(state, engine, actions)
    return Sample(
        board=board,
        side=side,
        legal_mask=mask,
        action_index=action_index(action),
        player=state["currentPlayer"],
    )


def select_model_action(
    model: PolicyValueNet,
    state: Dict[str, Any],
    engine: RustEngine,
    actions: List[Action],
    temperature: float = 1.0,
    device: torch.device | str = "cpu",
) -> Tuple[Action, Sample]:
    board, side, mask = encode_state(state, engine, actions)
    board_tensor = torch.tensor(board, dtype=torch.float32, device=device).unsqueeze(0)
    side_tensor = torch.tensor(side, dtype=torch.float32, device=device).unsqueeze(0)
    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        logits, _value = model(board_tensor, side_tensor)
        masked = masked_policy_logits(logits, mask_tensor).squeeze(0)
        if temperature <= 0:
            selected_index = int(masked.argmax().item())
        else:
            probs = torch.softmax(masked / temperature, dim=0)
            selected_index = int(torch.multinomial(probs, 1).item())

    selected = decode_action(selected_index)
    legal_indexes = {action_index(action) for action in actions}
    if selected_index not in legal_indexes:
        selected = actions[0]
        selected_index = action_index(selected)

    return selected, Sample(
        board=board,
        side=side,
        legal_mask=mask,
        action_index=selected_index,
        player=state["currentPlayer"],
    )


def play_game(
    engine: RustEngine,
    max_plies: int = 160,
    seed: int | None = None,
    initial_state: Dict[str, Any] | None = None,
    policy: PolicyMode = "random",
    model: PolicyValueNet | None = None,
    temperature: float = 1.0,
    device: torch.device | str = "cpu",
    cap_value: CapValueMode = "material",
    collect_metrics: bool = True,
    heuristic_variety: float = 0.55,
    heuristic_time_budget_ms: int = 10,
) -> GameRecord:
    if policy == "model" and model is None:
        raise ValueError("model policy requires a model")

    rng = random.Random(seed)
    state = initial_state or load_initial_state()
    samples: List[Sample] = []
    stats = GameStats()

    if model is not None:
        model.eval()

    for ply in range(max_plies):
        if state["status"] != "playing":
            break

        actions = engine.legal_actions(state)
        if not actions:
            break

        current_player = state["currentPlayer"]
        before_pieces = _piece_map(state)
        before_unstable = _unstable_ids_for_player(state, current_player)
        before_counts = _piece_counts(state)

        if collect_metrics:
            for player in ("red", "blue"):
                stats.pressure_sum[player] += len(engine.player_actions(state, player))
            stats.pressure_samples += 1
            if before_unstable:
                stats.rescue_opportunities += 1

        if policy == "model":
            assert model is not None
            action, sample = select_model_action(
                model,
                state,
                engine,
                actions,
                temperature=temperature,
                device=device,
            )
            samples.append(sample)
            next_state = engine.apply_action(state, action, analyze_checkmate=False)
        elif policy == "heuristic":
            next_state = engine.play_heuristic_turn(
                state,
                player=current_player,
                seed=(seed or 0) + ply,
                variety=heuristic_variety,
                time_budget_ms=heuristic_time_budget_ms,
            )
        else:
            action = rng.choice(actions)
            samples.append(_sample_from_action(state, engine, actions, action))
            next_state = engine.apply_action(state, action, analyze_checkmate=False)

        after_pieces = _piece_map(next_state)
        lost_ids = [piece_id for piece_id in before_pieces if piece_id not in after_pieces]
        for piece_id in lost_ids:
            lost_piece = before_pieces[piece_id]
            owner = lost_piece["owner"]
            piece_type = lost_piece["type"]
            stats.losses_by_player[owner] += 1
            stats.losses_by_piece_type[piece_type] = stats.losses_by_piece_type.get(piece_type, 0) + 1
            if stats.first_loss_player is None:
                stats.first_loss_player = owner
                stats.first_loss_piece_type = piece_type

        if collect_metrics and before_unstable:
            after_unstable = _unstable_ids_for_player(next_state, current_player)
            if before_unstable.isdisjoint(set(lost_ids)) and before_unstable.isdisjoint(after_unstable):
                stats.rescues += 1

        winner = _winner(next_state["status"])
        if winner is not None:
            loser = "blue" if winner == "red" else "red"
            stats.min_winner_pieces = min(before_counts[winner], _piece_counts(next_state)[winner])
            stats.max_loser_pieces = max(before_counts[loser], _piece_counts(next_state)[loser])

        stats.plies = ply + 1
        state = next_state

    stats.status = state["status"]
    stats.winner = _winner(state["status"])
    stats.final_piece_counts = _piece_counts(state)
    _assign_values(samples, state, cap_value)
    return GameRecord(samples=samples, stats=stats, final_state=state)


def random_selfplay_game(
    engine: RustEngine,
    max_plies: int = 160,
    seed: int | None = None,
    initial_state: Dict[str, Any] | None = None,
) -> List[Sample]:
    return play_game(
        engine,
        max_plies=max_plies,
        seed=seed,
        initial_state=initial_state,
        policy="random",
    ).samples


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


def selfplay_records(
    engine: RustEngine,
    games: int,
    max_plies: int = 160,
    seed: int = 0,
    policy: PolicyMode = "random",
    model: PolicyValueNet | None = None,
    temperature: float = 1.0,
    device: torch.device | str = "cpu",
    cap_value: CapValueMode = "material",
    collect_metrics: bool = True,
) -> List[GameRecord]:
    return [
        play_game(
            engine,
            max_plies=max_plies,
            seed=seed + game,
            policy=policy,
            model=model,
            temperature=temperature,
            device=device,
            cap_value=cap_value,
            collect_metrics=collect_metrics,
        )
        for game in range(games)
    ]
