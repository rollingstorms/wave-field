from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .encoding import ACTION_SIZE, BOARD_CHANNELS, BOARD_SIZE, action_index, decode_action, encode_state
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
    record_sample: bool = True,
) -> Tuple[Action, Sample | None]:
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

    sample = None
    if record_sample:
        sample = Sample(
            board=board,
            side=side,
            legal_mask=mask,
            action_index=selected_index,
            player=state["currentPlayer"],
        )

    return selected, sample


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
    device: torch.device | str = "cpu",
    cap_value: CapValueMode = "material",
    collect_metrics: bool = True,
    record_samples: bool = True,
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
                record_sample=record_samples,
            )
            if sample is not None:
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
            if record_samples:
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


def rust_random_training_samples(
    engine: RustEngine,
    games: int,
    max_plies: int = 160,
    seed: int = 0,
    cap_value: CapValueMode = "material",
) -> tuple[List[Sample], Dict[str, Any]]:
    batch = engine.generate_random_training_batch(
        load_initial_state(),
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
    device: torch.device | str = "cpu",
    cap_value: CapValueMode = "material",
    collect_metrics: bool = True,
    record_samples: bool = True,
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
            record_samples=record_samples,
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
                    active.discard(game_index)
                    continue
                encoded_indexes.append(game_index)
                batch_states.append(state)
                batch_actions.append(actions)
                encode_started_at = time.perf_counter()
                encoded.append(encode_state(state, engine, actions))
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


def _sample_from_rollout_position(position: Dict[str, Any]) -> Sample:
    legal_mask = np.zeros((ACTION_SIZE,), dtype=np.float32)
    legal_mask[np.asarray(position["legalActionIndexes"], dtype=np.int64)] = 1.0
    return Sample(
        board=np.asarray(position["board"], dtype=np.float32).reshape(
            BOARD_CHANNELS,
            BOARD_SIZE,
            BOARD_SIZE,
        ),
        side=np.asarray(position["side"], dtype=np.float32),
        legal_mask=legal_mask,
        action_index=-1,
        player=position["player"],
    )


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
    profile: Profile | None = None,
) -> List[GameRecord]:
    if games <= 0:
        return []

    model.eval()
    torch.manual_seed(seed)
    _profile_increment(profile, "games", games)
    started_at = time.perf_counter()

    create_started_at = time.perf_counter()
    session_id = engine.create_rollout_session(
        [load_initial_state() for _ in range(games)],
        max_plies=max_plies,
    )
    _profile_add(profile, "create_session_seconds", time.perf_counter() - create_started_at)
    sample_lists: List[List[Sample]] = [[] for _ in range(games)]

    try:
        while True:
            get_batch_started_at = time.perf_counter()
            rollout_batch = engine.get_rollout_batch(session_id)
            _profile_add(profile, "get_batch_seconds", time.perf_counter() - get_batch_started_at)
            positions = rollout_batch["positions"]
            if not positions:
                break

            pending_actions: List[Dict[str, int]] = []
            for start in range(0, len(positions), batch_size):
                chunk = positions[start:start + batch_size]
                tensor_started_at = time.perf_counter()
                samples = [_sample_from_rollout_position(position) for position in chunk]
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
        stats = GameStats(
            plies=int(game["plies"]),
            status=state["status"],
            winner=_winner(state["status"]),
            final_piece_counts=_piece_counts(state),
        )
        _assign_values(sample_lists[game_index], state, cap_value)
        records.append(GameRecord(samples=sample_lists[game_index], stats=stats, final_state=state))
    return records
