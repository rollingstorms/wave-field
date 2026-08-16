from __future__ import annotations

import random
import time
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .encoding import (
    ACTION_SIZE,
    BLUE_UNSTABLE_CHANNEL,
    BOARD_CHANNELS,
    BOARD_SIZE,
    CURRENT_PLAYER_CHANNEL,
    FIELD_MAGNITUDE_CHANNEL,
    FIELD_SIGNED_CHANNEL,
    PIECE_IDS,
    PIECE_TYPES,
    PLAYERS,
    RED_UNSTABLE_CHANNEL,
    InputView,
    action_index,
    board_channels_for_view,
    decode_action,
    decode_tuning_action,
    encode_state,
    legal_tuning_actions,
    legal_tuning_mask,
    tuning_action_index,
)
from .engine import Action, TuningAction, RustEngine, load_initial_state
from .model import PolicyValueNet, masked_policy_logits


Player = Literal["red", "blue"]
PolicyMode = Literal["random", "model", "heuristic", "easy"]
CapValueMode = Literal["zero", "material"]


@dataclass
class Sample:
    board: np.ndarray
    side: np.ndarray
    legal_mask: np.ndarray
    action_index: int
    player: str
    value: float = 0.0
    action_kind: int = 0
    legal_tuning_mask: np.ndarray | None = None
    tuning_action_index: int = -100
    history_board: np.ndarray | None = None
    history_side: np.ndarray | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    ai_turns_by_player: Dict[str, int] = field(default_factory=lambda: {"red": 0, "blue": 0})
    tune_turns_by_player: Dict[str, int] = field(default_factory=lambda: {"red": 0, "blue": 0})
    tune_actions_by_player: Dict[str, int] = field(default_factory=lambda: {"red": 0, "blue": 0})
    effective_tune_changes_by_player: Dict[str, int] = field(default_factory=lambda: {"red": 0, "blue": 0})
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
            "ai_turns_by_player": self.ai_turns_by_player,
            "tune_turns_by_player": self.tune_turns_by_player,
            "tune_actions_by_player": self.tune_actions_by_player,
            "effective_tune_changes_by_player": self.effective_tune_changes_by_player,
            "min_winner_pieces": self.min_winner_pieces,
            "max_loser_pieces": self.max_loser_pieces,
            "final_piece_counts": self.final_piece_counts,
        }


@dataclass
class GameRecord:
    samples: List[Sample]
    stats: GameStats
    final_state: Dict[str, Any]


Profile = MutableMapping[str, float]


def _profile_add(profile: Profile | None, key: str, value: float) -> None:
    if profile is not None:
        profile[key] = profile.get(key, 0.0) + value


def _profile_increment(profile: Profile | None, key: str, value: int = 1) -> None:
    if profile is not None:
        profile[key] = profile.get(key, 0.0) + float(value)


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


def _no_move_loss(state: Dict[str, Any]) -> Dict[str, Any]:
    winner = "red" if state["currentPlayer"] == "blue" else "blue"
    next_state = copy.deepcopy(state)
    next_state["status"] = f"{winner}-won"
    next_state["selectedPieceId"] = None
    next_state["message"] = f"{state['currentPlayer'].capitalize()} has no legal move"
    return next_state


def _piece_map(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {piece["id"]: piece for piece in state["pieces"]}


def _piece_counts(state: Dict[str, Any]) -> Dict[str, int]:
    counts = {"red": 0, "blue": 0}
    for piece in state["pieces"]:
        counts[piece["owner"]] += 1
    return counts


def _piece_type_counts_from_slots(pieces: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = {piece_type: 0 for piece_type in PIECE_TYPES}
    for piece in pieces:
        piece_id = PIECE_IDS[int(piece["slot"])]
        _owner, piece_type, *_rest = piece_id.split("-")
        counts[piece_type] += 1
    return counts


def _owner_counts_from_slots(pieces: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"red": 0, "blue": 0}
    for piece in pieces:
        piece_id = PIECE_IDS[int(piece["slot"])]
        owner, _piece_type, *_rest = piece_id.split("-")
        counts[owner] += 1
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


def _component_change_count(before: Dict[str, Any], after: Dict[str, Any], player: str) -> int:
    before_components = before["components"][player]
    after_components = after["components"][player]
    return sum(
        1
        for piece_type in PIECE_TYPES
        for left, right in zip(before_components[piece_type], after_components[piece_type])
        if left != right
    )


def _update_tuning_stats(stats: GameStats, player: str, tune_actions: int, effective_changes: int) -> None:
    stats.ai_turns_by_player[player] += 1
    stats.tune_actions_by_player[player] += tune_actions
    stats.effective_tune_changes_by_player[player] += effective_changes
    if tune_actions > 0:
        stats.tune_turns_by_player[player] += 1


def _annotate_turn_samples(
    samples: List[Sample],
    state: Dict[str, Any],
    ply: int,
    scenario: str | None = None,
) -> None:
    scenario = scenario or str(state.get("metadata", {}).get("scenario", "initial"))
    phase = _phase_for_ply(ply)
    for sample in samples:
        sample.metadata.setdefault("scenario", scenario)
        sample.metadata.setdefault("phase", phase)


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
    input_view: InputView = "base",
) -> Sample:
    board, side, mask = encode_state(state, engine, actions, input_view=input_view)
    return Sample(
        board=board,
        side=side,
        legal_mask=mask,
        action_index=action_index(action),
        player=state["currentPlayer"],
        metadata={
            "source": "python_rollout",
            "legal_count": len(actions),
        },
    )


EncodedHistory = Sequence[Tuple[np.ndarray, np.ndarray]]


def _history_arrays(
    current_board: np.ndarray,
    current_side: np.ndarray,
    history: EncodedHistory | None,
    history_plies: int,
) -> Tuple[np.ndarray, np.ndarray] | None:
    if history_plies <= 1:
        return None
    prior = list(history or [])
    window = [*prior, (current_board, current_side)][-history_plies:]
    pad_count = history_plies - len(window)
    board_pad = [
        np.zeros_like(current_board)
        for _ in range(pad_count)
    ]
    side_pad = [
        np.zeros_like(current_side)
        for _ in range(pad_count)
    ]
    history_boards = [*board_pad, *(item[0] for item in window)]
    history_sides = [*side_pad, *(item[1] for item in window)]
    return np.stack(history_boards).astype(np.float32), np.stack(history_sides).astype(np.float32)


def _attach_history(
    sample: Sample,
    board: np.ndarray,
    side: np.ndarray,
    history: EncodedHistory | None,
    history_plies: int,
) -> Sample:
    arrays = _history_arrays(board, side, history, history_plies)
    if arrays is not None:
        sample.history_board, sample.history_side = arrays
        sample.metadata["history_plies"] = history_plies
    return sample


def select_model_action(
    model: PolicyValueNet,
    state: Dict[str, Any],
    engine: RustEngine,
    actions: List[Action],
    temperature: float = 1.0,
    device: torch.device | str = "cpu",
    record_sample: bool = True,
    input_view: InputView = "base",
    history: EncodedHistory | None = None,
    history_plies: int = 1,
) -> Tuple[Action, Sample | None]:
    board, side, mask = encode_state(state, engine, actions, input_view=input_view)
    board_tensor = torch.tensor(board, dtype=torch.float32, device=device).unsqueeze(0)
    side_tensor = torch.tensor(side, dtype=torch.float32, device=device).unsqueeze(0)
    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
    history_arrays = _history_arrays(board, side, history, history_plies)
    history_board_tensor = (
        torch.tensor(history_arrays[0], dtype=torch.float32, device=device).unsqueeze(0)
        if history_arrays is not None
        else None
    )
    history_side_tensor = (
        torch.tensor(history_arrays[1], dtype=torch.float32, device=device).unsqueeze(0)
        if history_arrays is not None
        else None
    )

    with torch.no_grad():
        logits, _value = model(
            board_tensor,
            side_tensor,
            history_board=history_board_tensor,
            history_side=history_side_tensor,
        )
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

    sample = None
    if record_sample:
            sample = Sample(
                board=board,
                side=side,
                legal_mask=mask,
                action_index=selected_index,
                player=state["currentPlayer"],
                metadata={
                    "source": "python_model_rollout",
                    "legal_count": len(actions),
                },
            )
            _attach_history(sample, board, side, history, history_plies)

    return selected, sample


def _sample_from_tuning_action(
    state: Dict[str, Any],
    engine: RustEngine,
    move_actions: List[Action],
    tune_actions: List[TuningAction],
    action: TuningAction,
    input_view: InputView = "base",
    history: EncodedHistory | None = None,
    history_plies: int = 1,
) -> Sample:
    board, side, move_mask = encode_state(state, engine, move_actions, input_view=input_view)
    sample = Sample(
        board=board,
        side=side,
        legal_mask=move_mask,
        action_index=-100,
        player=state["currentPlayer"],
        action_kind=1,
        legal_tuning_mask=legal_tuning_mask(tune_actions),
        tuning_action_index=tuning_action_index(action),
        metadata={
            "source": "python_model_full_policy",
            "legal_count": len(move_actions),
            "legal_tuning_count": len(tune_actions),
        },
    )
    return _attach_history(sample, board, side, history, history_plies)


def _sample_from_model_move_action(
    state: Dict[str, Any],
    engine: RustEngine,
    move_actions: List[Action],
    action: Action,
    input_view: InputView = "base",
    history: EncodedHistory | None = None,
    history_plies: int = 1,
) -> Sample:
    sample = _sample_from_action(state, engine, move_actions, action, input_view=input_view)
    sample.action_kind = 0
    sample.tuning_action_index = -100
    sample.legal_tuning_mask = legal_tuning_mask(legal_tuning_actions(state))
    sample.metadata["source"] = "python_model_full_policy"
    sample.metadata["legal_tuning_count"] = int(sample.legal_tuning_mask.sum())
    return _attach_history(sample, sample.board, sample.side, history, history_plies)


def _heuristic_tuning_samples(
    state: Dict[str, Any],
    target_state: Dict[str, Any],
    engine: RustEngine,
    input_view: InputView = "base",
    history: EncodedHistory | None = None,
    history_plies: int = 1,
) -> Tuple[Dict[str, Any], List[Sample]]:
    player = state["currentPlayer"]
    probe = state
    samples: List[Sample] = []
    target_components = target_state["components"][player]
    for piece_type in PIECE_TYPES:
        for component_index, target_value in enumerate(target_components[piece_type]):
            if target_value == 0 or probe["components"][player][piece_type][component_index] == target_value:
                continue
            action: TuningAction = {
                "type": "tune",
                "pieceType": piece_type,
                "componentIndex": component_index,
                "value": int(target_value),
            }
            tune_actions = legal_tuning_actions(probe)
            legal_indexes = {tuning_action_index(candidate) for candidate in tune_actions}
            if tuning_action_index(action) not in legal_indexes:
                continue
            move_actions = engine.legal_actions(probe)
            if move_actions:
                sample = _sample_from_tuning_action(
                    probe,
                    engine,
                    move_actions,
                    tune_actions,
                    action,
                    input_view=input_view,
                    history=history,
                    history_plies=history_plies,
                )
                sample.metadata["source"] = "heuristic_bootstrap"
                samples.append(sample)
            probe = engine.apply_tuning(probe, action)
    return probe, samples


def _heuristic_move_action(before: Dict[str, Any], after: Dict[str, Any], player: str) -> Action | None:
    after_by_id = _piece_map(after)
    for piece in before["pieces"]:
        if piece["owner"] != player or piece["id"] not in after_by_id:
            continue
        after_piece = after_by_id[piece["id"]]
        if after_piece["position"] != piece["position"]:
            return {"pieceId": piece["id"], "destination": after_piece["position"]}
    return None


def heuristic_bootstrap_game(
    engine: RustEngine,
    max_plies: int = 160,
    seed: int | None = None,
    initial_state: Dict[str, Any] | None = None,
    cap_value: CapValueMode = "material",
    input_view: InputView = "base",
    heuristic_variety: float = 0.55,
    heuristic_time_budget_ms: int = 10,
    collect_metrics: bool = True,
    history_plies: int = 1,
) -> GameRecord:
    state = initial_state or load_initial_state()
    samples: List[Sample] = []
    stats = GameStats()
    encoded_history: List[Tuple[np.ndarray, np.ndarray]] = []
    scenario = str(state.get("metadata", {}).get("scenario", "initial"))

    for ply in range(max_plies):
        if state["status"] != "playing":
            break
        current_player = state["currentPlayer"]
        history_actions = engine.legal_actions(state)
        history_board, history_side, _history_mask = encode_state(
            state,
            engine,
            history_actions,
            input_view=input_view,
        )
        before_pieces = _piece_map(state)
        before_unstable = _unstable_ids_for_player(state, current_player)
        before_counts = _piece_counts(state)
        if collect_metrics:
            for player in ("red", "blue"):
                stats.pressure_sum[player] += len(engine.player_actions(state, player))
            stats.pressure_samples += 1
            if before_unstable:
                stats.rescue_opportunities += 1

        heuristic_state = engine.play_heuristic_turn(
            state,
            player=current_player,
            seed=(seed or 0) + ply,
            variety=heuristic_variety,
            time_budget_ms=heuristic_time_budget_ms,
        )
        tuned_state, tune_samples = _heuristic_tuning_samples(
            state,
            heuristic_state,
            engine,
            input_view=input_view,
            history=encoded_history,
            history_plies=history_plies,
        )
        samples.extend(tune_samples)
        _annotate_turn_samples(tune_samples, state, ply, scenario=scenario)
        move_action = _heuristic_move_action(tuned_state, heuristic_state, current_player)
        if move_action is not None:
            move_actions = engine.legal_actions(tuned_state)
            legal_indexes = {action_index(action) for action in move_actions}
            if action_index(move_action) in legal_indexes:
                sample = _sample_from_model_move_action(
                    tuned_state,
                    engine,
                    move_actions,
                    move_action,
                    input_view=input_view,
                    history=encoded_history,
                    history_plies=history_plies,
                )
                sample.metadata["source"] = "heuristic_bootstrap"
                _annotate_turn_samples([sample], state, ply, scenario=scenario)
                samples.append(sample)

        _update_tuning_stats(
            stats,
            current_player,
            len(tune_samples),
            _component_change_count(state, heuristic_state, current_player),
        )

        after_pieces = _piece_map(heuristic_state)
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
            after_unstable = _unstable_ids_for_player(heuristic_state, current_player)
            if before_unstable.isdisjoint(set(lost_ids)) and before_unstable.isdisjoint(after_unstable):
                stats.rescues += 1

        winner = _winner(heuristic_state["status"])
        if winner is not None:
            loser = "blue" if winner == "red" else "red"
            stats.min_winner_pieces = min(before_counts[winner], _piece_counts(heuristic_state)[winner])
            stats.max_loser_pieces = max(before_counts[loser], _piece_counts(heuristic_state)[loser])

        stats.plies = ply + 1
        encoded_history.append((history_board, history_side))
        if len(encoded_history) >= history_plies:
            encoded_history = encoded_history[-(history_plies - 1):] if history_plies > 1 else []
        state = heuristic_state

    stats.status = state["status"]
    stats.winner = _winner(state["status"])
    stats.final_piece_counts = _piece_counts(state)
    _assign_values(samples, state, cap_value)
    return GameRecord(samples=samples, stats=stats, final_state=state)


def heuristic_bootstrap_records(
    engine: RustEngine,
    games: int,
    max_plies: int = 160,
    seed: int = 0,
    cap_value: CapValueMode = "material",
    input_view: InputView = "base",
    initial_states: List[Dict[str, Any]] | None = None,
    collect_metrics: bool = True,
    history_plies: int = 1,
) -> List[GameRecord]:
    if initial_states is not None and len(initial_states) != games:
        raise ValueError("initial_states length must match games")
    return [
        heuristic_bootstrap_game(
            engine,
            max_plies=max_plies,
            seed=seed + game,
            initial_state=initial_states[game] if initial_states is not None else None,
            cap_value=cap_value,
            input_view=input_view,
            collect_metrics=collect_metrics,
            history_plies=history_plies,
        )
        for game in range(games)
    ]


def _select_index(masked: torch.Tensor, temperature: float) -> int:
    if temperature <= 0:
        return int(masked.argmax().item())
    probs = torch.softmax(masked / temperature, dim=0)
    return int(torch.multinomial(probs, 1).item())


def select_model_full_turn(
    model: PolicyValueNet,
    state: Dict[str, Any],
    engine: RustEngine,
    temperature: float = 1.0,
    kind_temperature: float | None = None,
    tuning_temperature: float | None = None,
    force_first_tune_prob: float = 0.0,
    rng: random.Random | None = None,
    device: torch.device | str = "cpu",
    record_samples: bool = True,
    input_view: InputView = "base",
    max_tuning_actions: int = 3,
    history: EncodedHistory | None = None,
    history_plies: int = 1,
) -> Tuple[Dict[str, Any], List[Sample], int]:
    samples: List[Sample] = []
    tuned_state = state
    tune_count = 0
    kind_temp = temperature if kind_temperature is None else kind_temperature
    tune_temp = temperature if tuning_temperature is None else tuning_temperature
    chooser = rng or random.Random()

    while True:
        move_actions = engine.legal_actions(tuned_state)
        tune_actions = legal_tuning_actions(tuned_state)
        can_tune = bool(tune_actions) and tune_count < max_tuning_actions
        if not move_actions and not can_tune:
            return tuned_state, samples, tune_count

        board, side, move_mask = encode_state(tuned_state, engine, move_actions, input_view=input_view)
        board_tensor = torch.tensor(board, dtype=torch.float32, device=device).unsqueeze(0)
        side_tensor = torch.tensor(side, dtype=torch.float32, device=device).unsqueeze(0)
        move_mask_tensor = torch.tensor(move_mask, dtype=torch.float32, device=device).unsqueeze(0)
        tune_mask = legal_tuning_mask(tune_actions)
        tune_mask_tensor = torch.tensor(tune_mask, dtype=torch.float32, device=device).unsqueeze(0)
        history_arrays = _history_arrays(board, side, history, history_plies)
        history_board_tensor = (
            torch.tensor(history_arrays[0], dtype=torch.float32, device=device).unsqueeze(0)
            if history_arrays is not None
            else None
        )
        history_side_tensor = (
            torch.tensor(history_arrays[1], dtype=torch.float32, device=device).unsqueeze(0)
            if history_arrays is not None
            else None
        )

        with torch.no_grad():
            kind_logits, move_logits, tune_logits = model.full_policy(
                board_tensor,
                side_tensor,
                history_board=history_board_tensor,
                history_side=history_side_tensor,
            )
            kind_mask = torch.tensor([[1.0 if move_actions else 0.0, 1.0 if can_tune else 0.0]], dtype=torch.float32, device=device)
            force_tune = can_tune and tune_count == 0 and force_first_tune_prob > 0 and chooser.random() < force_first_tune_prob
            kind = 1 if force_tune else _select_index(masked_policy_logits(kind_logits, kind_mask).squeeze(0), kind_temp)

            if kind == 1 and can_tune:
                selected_index = _select_index(masked_policy_logits(tune_logits, tune_mask_tensor).squeeze(0), tune_temp)
                legal_indexes = {tuning_action_index(action) for action in tune_actions}
                if selected_index not in legal_indexes:
                    selected_index = tuning_action_index(tune_actions[0])
                tune_action = decode_tuning_action(selected_index)
                if record_samples:
                    samples.append(
                        _sample_from_tuning_action(
                            tuned_state,
                            engine,
                            move_actions,
                            tune_actions,
                            tune_action,
                            input_view=input_view,
                            history=history,
                            history_plies=history_plies,
                        )
                    )
                tuned_state = engine.apply_tuning(tuned_state, tune_action)
                tune_count += 1
                continue

            selected_index = _select_index(masked_policy_logits(move_logits, move_mask_tensor).squeeze(0), temperature)
            legal_indexes = {action_index(action) for action in move_actions}
            if selected_index in legal_indexes:
                move_action = decode_action(selected_index)
            else:
                move_action = move_actions[0]
                selected_index = action_index(move_action)
            if record_samples:
                sample = _sample_from_model_move_action(tuned_state, engine, move_actions, move_action, input_view=input_view)
                _attach_history(sample, sample.board, sample.side, history, history_plies)
                sample.action_index = selected_index
                samples.append(sample)
            return engine.apply_action(tuned_state, move_action, analyze_checkmate=False), samples, tune_count


def select_model_actions_batched(
    model: PolicyValueNet,
    states: Sequence[Dict[str, Any]],
    encoded: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    actions_by_state: Sequence[List[Action]],
    temperature: float = 1.0,
    device: torch.device | str = "cpu",
    record_samples: bool = True,
) -> List[Tuple[Action, Sample | None]]:
    boards = torch.tensor(np.stack([item[0] for item in encoded]), dtype=torch.float32, device=device)
    sides = torch.tensor(np.stack([item[1] for item in encoded]), dtype=torch.float32, device=device)
    masks = torch.tensor(np.stack([item[2] for item in encoded]), dtype=torch.float32, device=device)

    with torch.no_grad():
        logits, _values = model(boards, sides)
        masked = masked_policy_logits(logits, masks)
        if temperature <= 0:
            selected_indexes = masked.argmax(dim=1).tolist()
        else:
            probs = torch.softmax(masked / temperature, dim=1)
            selected_indexes = torch.multinomial(probs, 1).squeeze(1).tolist()

    selections: List[Tuple[Action, Sample | None]] = []
    for state, (board, side, mask), actions, selected_index in zip(states, encoded, actions_by_state, selected_indexes):
        legal_indexes = {action_index(action) for action in actions}
        if selected_index in legal_indexes:
            action = decode_action(int(selected_index))
        else:
            action = actions[0]
            selected_index = action_index(action)

        sample = None
        if record_samples:
            sample = Sample(
                board=board,
                side=side,
                legal_mask=mask,
                action_index=int(selected_index),
                player=state["currentPlayer"],
                metadata={
                    "source": "python_batched_model_rollout",
                    "legal_count": len(actions),
                },
            )
        selections.append((action, sample))

    return selections


def play_game(
    engine: RustEngine,
    max_plies: int = 160,
    seed: int | None = None,
    initial_state: Dict[str, Any] | None = None,
    policy: PolicyMode = "random",
    model: PolicyValueNet | None = None,
    temperature: float = 1.0,
    kind_temperature: float | None = None,
    tuning_temperature: float | None = None,
    force_first_tune_prob: float = 0.0,
    device: torch.device | str = "cpu",
    cap_value: CapValueMode = "material",
    collect_metrics: bool = True,
    record_samples: bool = True,
    heuristic_variety: float = 0.55,
    heuristic_time_budget_ms: int = 10,
    input_view: InputView = "base",
    full_policy: bool = False,
    max_tuning_actions: int = 3,
    history_plies: int = 1,
) -> GameRecord:
    if policy == "model" and model is None:
        raise ValueError("model policy requires a model")

    rng = random.Random(seed)
    state = initial_state or load_initial_state()
    samples: List[Sample] = []
    stats = GameStats()
    encoded_history: List[Tuple[np.ndarray, np.ndarray]] = []
    scenario = str(state.get("metadata", {}).get("scenario", "initial"))

    if model is not None:
        model.eval()

    for ply in range(max_plies):
        if state["status"] != "playing":
            break

        actions = engine.legal_actions(state)
        if not actions and not full_policy:
            state = _no_move_loss(state)
            stats.plies = ply + 1
            break
        history_actions = actions if actions else engine.legal_actions(state)
        history_board, history_side, _history_mask = encode_state(state, engine, history_actions, input_view=input_view)

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
            if full_policy:
                next_state, turn_samples, tune_actions = select_model_full_turn(
                    model,
                    state,
                    engine,
                    temperature=temperature,
                    kind_temperature=kind_temperature,
                    tuning_temperature=tuning_temperature,
                    force_first_tune_prob=force_first_tune_prob,
                    rng=rng,
                    device=device,
                    record_samples=record_samples,
                    input_view=input_view,
                    max_tuning_actions=max_tuning_actions,
                    history=encoded_history,
                    history_plies=history_plies,
                )
                _annotate_turn_samples(turn_samples, state, ply, scenario=scenario)
                if next_state["status"] == "playing" and not engine.legal_actions(next_state):
                    next_state = _no_move_loss(next_state)
                samples.extend(turn_samples)
                _update_tuning_stats(
                    stats,
                    current_player,
                    tune_actions,
                    _component_change_count(state, next_state, current_player),
                )
            else:
                action, sample = select_model_action(
                    model,
                    state,
                    engine,
                    actions,
                    temperature=temperature,
                    device=device,
                    record_sample=record_samples,
                    input_view=input_view,
                    history=encoded_history,
                    history_plies=history_plies,
                )
                if sample is not None:
                    _annotate_turn_samples([sample], state, ply, scenario=scenario)
                    samples.append(sample)
                next_state = engine.apply_action(state, action, analyze_checkmate=False)
                _update_tuning_stats(stats, current_player, 0, 0)
        elif policy == "heuristic":
            next_state = engine.play_heuristic_turn(
                state,
                player=current_player,
                seed=(seed or 0) + ply,
                variety=heuristic_variety,
                time_budget_ms=heuristic_time_budget_ms,
            )
            tune_changes = _component_change_count(state, next_state, current_player)
            _update_tuning_stats(stats, current_player, tune_changes, tune_changes)
        elif policy == "easy":
            next_state = engine.play_easy_turn(
                state,
                player=current_player,
                seed=(seed or 0) + ply,
                variety=0.0,
                time_budget_ms=heuristic_time_budget_ms,
            )
            _update_tuning_stats(stats, current_player, 0, 0)
        else:
            action = rng.choice(actions)
            if record_samples:
                sample = _sample_from_action(state, engine, actions, action, input_view=input_view)
                _annotate_turn_samples([sample], state, ply, scenario=scenario)
                samples.append(sample)
            next_state = engine.apply_action(state, action, analyze_checkmate=False)
            _update_tuning_stats(stats, current_player, 0, 0)

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
        encoded_history.append((history_board, history_side))
        if len(encoded_history) >= history_plies:
            encoded_history = encoded_history[-(history_plies - 1):] if history_plies > 1 else []
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


def rust_random_training_samples(
    engine: RustEngine,
    games: int,
    max_plies: int = 160,
    seed: int = 0,
    cap_value: CapValueMode = "material",
    initial_state: Dict[str, Any] | None = None,
    metadata: Dict[str, Any] | None = None,
) -> tuple[List[Sample], Dict[str, Any]]:
    batch = engine.generate_random_training_batch(
        initial_state or load_initial_state(),
        games=games,
        max_plies=max_plies,
        seed=seed,
        material_for_capped=cap_value == "material",
    )
    samples = []
    for raw in batch["samples"]:
        legal_mask = np.zeros((ACTION_SIZE,), dtype=np.float32)
        legal_mask[np.asarray(raw["legalActionIndexes"], dtype=np.int64)] = 1.0
        samples.append(
            Sample(
                board=np.asarray(raw["board"], dtype=np.float32).reshape(
                    BOARD_CHANNELS,
                    BOARD_SIZE,
                    BOARD_SIZE,
                ),
                side=np.asarray(raw["side"], dtype=np.float32),
                legal_mask=legal_mask,
                action_index=int(raw["actionIndex"]),
                player=raw["player"],
                value=float(raw["value"]),
                metadata={
                    "source": "rust_random",
                    "legal_count": len(raw["legalActionIndexes"]),
                    **(metadata or {}),
                },
            )
        )
    return samples, batch["summary"]


def selfplay_records(
    engine: RustEngine,
    games: int,
    max_plies: int = 160,
    seed: int = 0,
    policy: PolicyMode = "random",
    model: PolicyValueNet | None = None,
    temperature: float = 1.0,
    kind_temperature: float | None = None,
    tuning_temperature: float | None = None,
    force_first_tune_prob: float = 0.0,
    device: torch.device | str = "cpu",
    cap_value: CapValueMode = "material",
    collect_metrics: bool = True,
    record_samples: bool = True,
    input_view: InputView = "base",
    full_policy: bool = False,
    max_tuning_actions: int = 3,
    history_plies: int = 1,
    initial_states: List[Dict[str, Any]] | None = None,
) -> List[GameRecord]:
    if initial_states is not None and len(initial_states) != games:
        raise ValueError("initial_states length must match games")
    return [
        play_game(
            engine,
            max_plies=max_plies,
            seed=seed + game,
            initial_state=initial_states[game] if initial_states is not None else None,
            policy=policy,
            model=model,
            temperature=temperature,
            kind_temperature=kind_temperature,
            tuning_temperature=tuning_temperature,
            force_first_tune_prob=force_first_tune_prob,
            device=device,
            cap_value=cap_value,
            collect_metrics=collect_metrics,
            record_samples=record_samples,
            input_view=input_view,
            full_policy=full_policy,
            max_tuning_actions=max_tuning_actions,
            history_plies=history_plies,
        )
        for game in range(games)
    ]


def batched_model_selfplay_records(
    engine: RustEngine,
    model: PolicyValueNet,
    games: int,
    max_plies: int = 160,
    seed: int = 0,
    temperature: float = 1.0,
    device: torch.device | str = "cpu",
    cap_value: CapValueMode = "material",
    batch_size: int = 32,
    record_samples: bool = True,
    profile: Profile | None = None,
    input_view: InputView = "base",
) -> List[GameRecord]:
    if games <= 0:
        return []

    model.eval()
    states = [load_initial_state() for _ in range(games)]
    sample_lists: List[List[Sample]] = [[] for _ in range(games)]
    stats = [GameStats() for _ in range(games)]
    active = set(range(games))
    _profile_increment(profile, "games", games)

    torch.manual_seed(seed)
    started_at = time.perf_counter()

    for ply in range(max_plies):
        active_indexes = [index for index in sorted(active) if states[index]["status"] == "playing"]
        if not active_indexes:
            break

        for start in range(0, len(active_indexes), batch_size):
            batch_indexes = active_indexes[start:start + batch_size]
            encoded_indexes: List[int] = []
            batch_states: List[Dict[str, Any]] = []
            batch_actions: List[List[Action]] = []
            encoded = []

            for game_index in batch_indexes:
                state = states[game_index]
                legal_started_at = time.perf_counter()
                actions = engine.legal_actions(state)
                _profile_add(profile, "legal_actions_seconds", time.perf_counter() - legal_started_at)
                if not actions:
                    states[game_index] = _no_move_loss(state)
                    stats[game_index].plies = ply + 1
                    active.discard(game_index)
                    continue
                encoded_indexes.append(game_index)
                batch_states.append(state)
                batch_actions.append(actions)
                encode_started_at = time.perf_counter()
                encoded.append(encode_state(state, engine, actions, input_view=input_view))
                _profile_add(profile, "encode_seconds", time.perf_counter() - encode_started_at)

            if not batch_states:
                continue

            inference_started_at = time.perf_counter()
            selections = select_model_actions_batched(
                model,
                batch_states,
                encoded,
                batch_actions,
                temperature=temperature,
                device=device,
                record_samples=record_samples,
            )
            _profile_add(profile, "inference_seconds", time.perf_counter() - inference_started_at)
            _profile_increment(profile, "batches")
            _profile_increment(profile, "positions", len(batch_states))

            for game_index, (action, sample) in zip(encoded_indexes, selections):
                before = states[game_index]
                before_pieces = _piece_map(before)
                apply_started_at = time.perf_counter()
                next_state = engine.apply_action(before, action, analyze_checkmate=False)
                _profile_add(profile, "apply_seconds", time.perf_counter() - apply_started_at)
                if sample is not None:
                    sample_lists[game_index].append(sample)
                    _profile_increment(profile, "samples")

                after_pieces = _piece_map(next_state)
                lost_ids = [piece_id for piece_id in before_pieces if piece_id not in after_pieces]
                for piece_id in lost_ids:
                    lost_piece = before_pieces[piece_id]
                    owner = lost_piece["owner"]
                    piece_type = lost_piece["type"]
                    stats[game_index].losses_by_player[owner] += 1
                    stats[game_index].losses_by_piece_type[piece_type] = (
                        stats[game_index].losses_by_piece_type.get(piece_type, 0) + 1
                    )
                    if stats[game_index].first_loss_player is None:
                        stats[game_index].first_loss_player = owner
                        stats[game_index].first_loss_piece_type = piece_type

                states[game_index] = next_state
                stats[game_index].plies = ply + 1
                if next_state["status"] != "playing":
                    active.discard(game_index)

    _profile_add(profile, "total_seconds", time.perf_counter() - started_at)
    records = []
    for game_index, state in enumerate(states):
        stats[game_index].status = state["status"]
        stats[game_index].winner = _winner(state["status"])
        stats[game_index].final_piece_counts = _piece_counts(state)
        _assign_values(sample_lists[game_index], state, cap_value)
        records.append(GameRecord(samples=sample_lists[game_index], stats=stats[game_index], final_state=state))
    return records


def _sample_from_rollout_position(position: Dict[str, Any], input_view: InputView = "base") -> Sample:
    legal_mask = np.zeros((ACTION_SIZE,), dtype=np.float32)
    legal_mask[np.asarray(position["legalActionIndexes"], dtype=np.int64)] = 1.0
    board = np.zeros((board_channels_for_view(input_view), BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    owner_counts = _owner_counts_from_slots(position["pieces"])
    piece_type_counts = _piece_type_counts_from_slots(position["pieces"])
    for piece in position["pieces"]:
        piece_id = PIECE_IDS[int(piece["slot"])]
        owner, piece_type, *_rest = piece_id.split("-")
        x = int(piece["x"])
        y = int(piece["y"])
        board[PLAYERS.index(owner) * len(PIECE_TYPES) + PIECE_TYPES.index(piece_type), y, x] = 1.0
        if input_view == "piece_identity":
            board[BOARD_CHANNELS + int(piece["slot"]), y, x] = 1.0
        if bool(piece["unstable"]):
            unstable_channel = RED_UNSTABLE_CHANNEL if owner == "red" else BLUE_UNSTABLE_CHANNEL
            board[unstable_channel, y, x] = 1.0

    field = np.asarray(position["field"], dtype=np.float32).reshape(BOARD_SIZE, BOARD_SIZE)
    board[FIELD_SIGNED_CHANNEL] = np.clip(field / 8.0, -1.0, 1.0)
    board[FIELD_MAGNITUDE_CHANNEL] = np.clip(np.abs(field) / 8.0, 0.0, 1.0)
    player = position["player"]
    opponent = "blue" if player == "red" else "red"
    total_pieces = owner_counts["red"] + owner_counts["blue"]
    board[CURRENT_PLAYER_CHANNEL].fill(1.0 if player == "red" else -1.0)

    return Sample(
        board=board,
        side=np.asarray(position["side"], dtype=np.float32),
        legal_mask=legal_mask,
        action_index=-1,
        player=player,
        metadata={
            "source": "rust_session_model",
            "game_index": int(position["gameIndex"]),
            "ply": int(position["ply"]),
            "phase": _phase_for_ply(int(position["ply"])),
            "legal_count": len(position["legalActionIndexes"]),
            "red_pieces": owner_counts["red"],
            "blue_pieces": owner_counts["blue"],
            "current_player_pieces": owner_counts[player],
            "opponent_pieces": owner_counts[opponent],
            "material_balance_current": owner_counts[player] - owner_counts[opponent],
            "total_pieces": total_pieces,
            "low_material": total_pieces <= 8,
            "piece_type_counts": piece_type_counts,
        },
    )


def _phase_for_ply(ply: int) -> str:
    if ply < 20:
        return "opening"
    if ply < 80:
        return "midgame"
    return "endgame"


def session_model_selfplay_records(
    engine: RustEngine,
    model: PolicyValueNet,
    games: int,
    max_plies: int = 160,
    seed: int = 0,
    temperature: float = 1.0,
    device: torch.device | str = "cpu",
    cap_value: CapValueMode = "material",
    batch_size: int = 32,
    record_samples: bool = True,
    collect_metrics: bool = False,
    profile: Profile | None = None,
    initial_states: List[Dict[str, Any]] | None = None,
    input_view: InputView = "base",
) -> List[GameRecord]:
    if games <= 0:
        return []
    if initial_states is not None and len(initial_states) != games:
        raise ValueError("initial_states length must match games")

    model.eval()
    torch.manual_seed(seed)
    _profile_increment(profile, "games", games)
    started_at = time.perf_counter()

    create_started_at = time.perf_counter()
    rollout_states = initial_states or [load_initial_state() for _ in range(games)]
    scenario_by_game = [
        str(state.get("metadata", {}).get("scenario", "initial"))
        for state in rollout_states
    ]
    session_id = engine.create_rollout_session(
        rollout_states,
        max_plies=max_plies,
        collect_pressure=collect_metrics,
    )
    _profile_add(profile, "create_session_seconds", time.perf_counter() - create_started_at)
    sample_lists: List[List[Sample]] = [[] for _ in range(games)]

    try:
        while True:
            get_batch_started_at = time.perf_counter()
            rollout_batch = engine.get_rollout_batch(session_id, profile=profile is not None)
            _profile_add(profile, "get_batch_seconds", time.perf_counter() - get_batch_started_at)
            rust_batch_profile = rollout_batch.get("profile")
            if profile is not None and rust_batch_profile:
                _profile_add(profile, "rust_batch_field_seconds", rust_batch_profile["fieldMs"] / 1000.0)
                _profile_add(
                    profile,
                    "rust_batch_legal_indexes_seconds",
                    rust_batch_profile["legalIndexesMs"] / 1000.0,
                )
                _profile_add(profile, "rust_batch_pieces_seconds", rust_batch_profile["piecesMs"] / 1000.0)
                _profile_add(
                    profile,
                    "rust_batch_flatten_field_seconds",
                    rust_batch_profile["flattenFieldMs"] / 1000.0,
                )
                _profile_add(profile, "rust_batch_side_seconds", rust_batch_profile["sideMs"] / 1000.0)
                _profile_increment(profile, "rust_batch_positions", int(rust_batch_profile["positions"]))
            positions = rollout_batch["positions"]
            if not positions:
                break

            pending_actions: List[Dict[str, int]] = []
            for start in range(0, len(positions), batch_size):
                chunk = positions[start:start + batch_size]
                tensor_started_at = time.perf_counter()
                samples = [_sample_from_rollout_position(position, input_view=input_view) for position in chunk]
                boards = torch.tensor(
                    np.stack([sample.board for sample in samples]),
                    dtype=torch.float32,
                    device=device,
                )
                sides = torch.tensor(
                    np.stack([sample.side for sample in samples]),
                    dtype=torch.float32,
                    device=device,
                )
                masks = torch.tensor(
                    np.stack([sample.legal_mask for sample in samples]),
                    dtype=torch.float32,
                    device=device,
                )
                _profile_add(profile, "tensor_prep_seconds", time.perf_counter() - tensor_started_at)

                inference_started_at = time.perf_counter()
                with torch.no_grad():
                    logits, _values = model(boards, sides)
                    masked = masked_policy_logits(logits, masks)
                    if temperature <= 0:
                        selected_indexes = masked.argmax(dim=1).tolist()
                    else:
                        probs = torch.softmax(masked / temperature, dim=1)
                        selected_indexes = torch.multinomial(probs, 1).squeeze(1).tolist()
                _profile_add(profile, "inference_seconds", time.perf_counter() - inference_started_at)
                _profile_increment(profile, "batches")
                _profile_increment(profile, "positions", len(chunk))

                selection_started_at = time.perf_counter()
                for position, sample, selected_index in zip(chunk, samples, selected_indexes):
                    legal_indexes = {int(index) for index in position["legalActionIndexes"]}
                    if int(selected_index) not in legal_indexes:
                        selected_index = int(position["legalActionIndexes"][0])
                    sample.action_index = int(selected_index)
                    game_index = int(position["gameIndex"])
                    sample.metadata["scenario"] = scenario_by_game[game_index]
                    if record_samples:
                        sample_lists[game_index].append(sample)
                        _profile_increment(profile, "samples")
                    pending_actions.append(
                        {
                            "gameIndex": game_index,
                            "actionIndex": int(selected_index),
                        }
                    )
                _profile_add(profile, "selection_seconds", time.perf_counter() - selection_started_at)

            if pending_actions:
                apply_started_at = time.perf_counter()
                engine.apply_rollout_actions(session_id, pending_actions)
                _profile_add(profile, "apply_batch_seconds", time.perf_counter() - apply_started_at)
                _profile_increment(profile, "apply_batches")

        finish_started_at = time.perf_counter()
        finished = engine.finish_rollout_session(session_id)
        _profile_add(profile, "finish_session_seconds", time.perf_counter() - finish_started_at)
    except Exception:
        try:
            engine.finish_rollout_session(session_id)
        except Exception:
            pass
        raise

    _profile_add(profile, "total_seconds", time.perf_counter() - started_at)
    records = []
    for game in sorted(finished["games"], key=lambda item: int(item["gameIndex"])):
        game_index = int(game["gameIndex"])
        state = game["state"]
        metrics = game.get("metrics", {})
        stats = GameStats(
            plies=int(game["plies"]),
            status=state["status"],
            winner=_winner(state["status"]),
            first_loss_player=metrics.get("firstLossPlayer"),
            first_loss_piece_type=metrics.get("firstLossPieceType"),
            losses_by_player={
                "red": int(metrics.get("lossesByPlayer", {}).get("red", 0)),
                "blue": int(metrics.get("lossesByPlayer", {}).get("blue", 0)),
            },
            losses_by_piece_type={
                piece_type: int(count)
                for piece_type, count in metrics.get("lossesByPieceType", {}).items()
            },
            rescue_opportunities=int(metrics.get("rescueOpportunities", 0)),
            rescues=int(metrics.get("rescues", 0)),
            pressure_sum={
                "red": int(metrics.get("pressureSum", {}).get("red", 0)),
                "blue": int(metrics.get("pressureSum", {}).get("blue", 0)),
            },
            pressure_samples=int(metrics.get("pressureSamples", 0)),
            final_piece_counts=_piece_counts(state),
        )
        _assign_values(sample_lists[game_index], state, cap_value)
        records.append(GameRecord(samples=sample_lists[game_index], stats=stats, final_state=state))
    return records
