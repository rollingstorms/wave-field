from __future__ import annotations

import math
import json
import multiprocessing
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch

from .encoding import (
    InputView,
    action_index,
    encode_state,
    legal_tuning_actions,
    legal_tuning_mask,
    tuning_action_index,
    tuning_limit_allows,
)
from .engine import Action, RustEngine, TuningAction, without_undo_history
from .model import PolicyValueNet, masked_policy_logits
from .selfplay import EncodedHistory, _history_arrays, result_value
from .selfplay import CapValueMode, GameRecord, GameStats, Sample, _assign_values, _piece_counts


SearchAction = Union[Action, TuningAction]


def repetition_key(state: Dict[str, Any]) -> str:
    """Rule position at a turn boundary, excluding UI and move counters."""
    position = {key: value for key, value in state.items() if key not in {
        'history', 'turnNumber', 'selectedPieceId', 'message', 'status', 'terminationReason',
    }}
    position['pieces'] = sorted(state['pieces'], key=lambda piece: piece['id'])
    return json.dumps(position, sort_keys=True, separators=(',', ':'))


def record_turn_tactics(stats: GameStats, before: Dict[str, Any], after: Dict[str, Any]) -> None:
    """Measure committed turns only; no engine queries or search-node accounting."""
    player = before['currentPlayer']
    old = {piece['id']: piece for piece in before['pieces']}
    new = {piece['id']: piece for piece in after['pieces']}
    endangered = {key for key, piece in old.items() if piece['owner'] == player and piece.get('unstable', False)}
    for key in (key for key in old if key not in new):
        piece = old[key]
        stats.losses_by_player[piece['owner']] += 1
        kind = piece['type']
        stats.losses_by_piece_type[kind] = stats.losses_by_piece_type.get(kind, 0) + 1
        if stats.first_loss_player is None:
            stats.first_loss_player, stats.first_loss_piece_type = piece['owner'], kind
    if endangered:
        stats.rescue_opportunities += 1
        stats.rescue_opportunities_by_player[player] += 1
        if all(key in new and not new[key].get('unstable', False) for key in endangered):
            stats.rescues += 1
            stats.rescues_by_player[player] += 1


def _opponent(player: str) -> str:
    return "blue" if player == "red" else "red"


def _no_move_loss(state: Dict[str, Any]) -> Dict[str, Any]:
    winner = _opponent(state["currentPlayer"])
    return {
        **state,
        "status": f"{winner}-won",
        "selectedPieceId": None,
        "message": f"{state['currentPlayer'].capitalize()} has no legal move",
    }


def _is_tune(action: SearchAction) -> bool:
    return action.get("type") == "tune"


def _action_key(action: SearchAction) -> str:
    if _is_tune(action):
        return f"tune:{action['pieceType']}:{action['componentIndex']}:{action['value']}"
    return f"move:{action['pieceId']}:{action['destination']['x']}:{action['destination']['y']}"


@dataclass
class MctsConfig:
    simulations: int = 128
    repetition_limit: int = 0
    value_discount: float = 1.0
    c_puct: float = 1.5
    top_k: int = 0
    min_tune_candidates: int = 1
    min_move_candidates: int = 1
    max_depth: int = 16
    max_tuning_actions: int = 3
    policy_temperature: float = 1.0
    dirichlet_alpha: float = 0.0
    exploration_fraction: float = 0.0
    force_first_tune_prob: float = 0.0
    force_tune_prob: float = 0.0
    sample_first_plies: int = 0


@dataclass
class SearchChild:
    action: SearchAction
    prior: float
    node: SearchNode | None = None


@dataclass
class SearchNode:
    state: Dict[str, Any]
    player: str
    tune_count: int
    prior: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    evaluation: float | None = None
    children: Dict[str, SearchChild] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class MctsResult:
    action: SearchAction
    visits: int
    value: float
    root_value: float
    policy: List[Dict[str, Any]]


def _model_outputs(
    model: PolicyValueNet,
    state: Dict[str, Any],
    engine: RustEngine,
    move_actions: Sequence[Action],
    tune_actions: Sequence[TuningAction],
    device: torch.device | str,
    input_view: InputView,
    history: EncodedHistory | None,
    history_plies: int,
    policy_temperature: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    board, side, move_mask = encode_state(state, engine, move_actions, input_view=input_view)
    board_tensor = torch.tensor(board, dtype=torch.float32, device=device).unsqueeze(0)
    side_tensor = torch.tensor(side, dtype=torch.float32, device=device).unsqueeze(0)
    move_mask_tensor = torch.tensor(move_mask, dtype=torch.float32, device=device).unsqueeze(0)
    tune_mask_tensor = torch.tensor(
        legal_tuning_mask(tune_actions),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
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
        kind_logits, move_logits, tune_logits, values = model.search_outputs(
            board_tensor,
            side_tensor,
            history_board=history_board_tensor,
            history_side=history_side_tensor,
        )
        kind_mask = torch.tensor(
            [[1.0 if move_actions else 0.0, 1.0 if tune_actions else 0.0]],
            dtype=torch.float32,
            device=device,
        )
        temperature = max(policy_temperature, 1.0e-6)
        kind_probs = torch.softmax(masked_policy_logits(kind_logits, kind_mask).squeeze(0) / temperature, dim=0)
        move_probs = torch.softmax(masked_policy_logits(move_logits, move_mask_tensor).squeeze(0) / temperature, dim=0)
        tune_probs = torch.softmax(masked_policy_logits(tune_logits, tune_mask_tensor).squeeze(0) / temperature, dim=0)
    return (
        (kind_probs[0] * move_probs).detach().cpu().numpy(),
        (kind_probs[1] * tune_probs).detach().cpu().numpy(),
        float(values.squeeze(0).item()),
    )


def _legal_search_actions(
    state: Dict[str, Any],
    engine: RustEngine,
    tune_count: int,
    max_tuning_actions: int,
) -> Tuple[List[Action], List[TuningAction], List[SearchAction]]:
    moves = engine.legal_actions(state)
    tunes = legal_tuning_actions(state) if tuning_limit_allows(tune_count, max_tuning_actions) else []
    return moves, tunes, [*tunes, *moves]


def _expand(
    node: SearchNode,
    model: PolicyValueNet,
    engine: RustEngine,
    config: MctsConfig,
    device: torch.device | str,
    input_view: InputView,
    history: EncodedHistory | None,
    history_plies: int,
    root_noise: bool = False,
    rng: np.random.Generator | None = None,
) -> float:
    if node.state["status"] != "playing":
        return result_value(node.state["status"], node.player)

    move_actions, tune_actions, actions = _legal_search_actions(
        node.state,
        engine,
        node.tune_count,
        config.max_tuning_actions,
    )
    if not actions:
        return -1.0

    move_priors, tune_priors, value = _model_outputs(
        model,
        node.state,
        engine,
        move_actions,
        tune_actions,
        device,
        input_view,
        history,
        history_plies,
        config.policy_temperature,
    )
    node.evaluation = value
    tune_scored: List[Tuple[float, SearchAction]] = [
        (float(tune_priors[tuning_action_index(action)]), action)
        for action in tune_actions
    ]
    move_scored: List[Tuple[float, SearchAction]] = [
        (float(move_priors[action_index(action)]), action)
        for action in move_actions
    ]
    tune_scored.sort(key=lambda item: item[0], reverse=True)
    move_scored.sort(key=lambda item: item[0], reverse=True)

    forced: Dict[str, Tuple[float, SearchAction]] = {}
    for prior, action in tune_scored[:max(0, config.min_tune_candidates)]:
        forced[_action_key(action)] = (prior, action)
    for prior, action in move_scored[:max(0, config.min_move_candidates)]:
        forced[_action_key(action)] = (prior, action)

    scored = sorted([*tune_scored, *move_scored], key=lambda item: item[0], reverse=True)
    selected: Dict[str, Tuple[float, SearchAction]] = dict(forced)
    for prior, action in scored:
        if config.top_k > 0 and len(selected) >= config.top_k:
            break
        selected.setdefault(_action_key(action), (prior, action))
    scored = sorted(selected.values(), key=lambda item: item[0], reverse=True)

    total_prior = sum(max(1.0e-8, prior) for prior, _action in scored)
    if total_prior <= 0:
        normalized = [(1.0 / len(scored), action) for _prior, action in scored]
    else:
        normalized = [(max(1.0e-8, prior) / total_prior, action) for prior, action in scored]

    if root_noise and config.dirichlet_alpha > 0 and config.exploration_fraction > 0 and len(normalized) > 1:
        noise = (rng or np.random.default_rng()).dirichlet([config.dirichlet_alpha] * len(normalized))
        normalized = [
            (
                prior * (1.0 - config.exploration_fraction)
                + float(noise[index]) * config.exploration_fraction,
                action,
            )
            for index, (prior, action) in enumerate(normalized)
        ]

    for prior, action in normalized:
        node.children[_action_key(action)] = SearchChild(action=action, prior=prior)
    return value


def _apply_search_action(state: Dict[str, Any], action: SearchAction, engine: RustEngine) -> Dict[str, Any]:
    if _is_tune(action):
        return without_undo_history(engine.apply_tuning(without_undo_history(state), action))  # type: ignore[arg-type]
    next_state = without_undo_history(engine.apply_action(without_undo_history(state), action, analyze_checkmate=False))  # type: ignore[arg-type]
    if next_state["status"] == "playing" and not engine.legal_actions(next_state):
        return _no_move_loss(next_state)
    return next_state


def _child_value_for_parent(parent: SearchNode, child: SearchNode, discount: float = 1.0) -> float:
    value = child.value if child.player == parent.player else -child.value
    return discount * value


def _select_child(node: SearchNode, config: MctsConfig) -> SearchChild:
    parent_visits = max(1, node.visits)
    best_score = -float("inf")
    best_child: SearchChild | None = None
    for child in node.children.values():
        q_value = _child_value_for_parent(node, child.node, 1.0 if _is_tune(child.action) else config.value_discount) if child.node is not None else 0.0
        visits = child.node.visits if child.node is not None else 0
        exploration = config.c_puct * child.prior * math.sqrt(parent_visits) / (1 + visits)
        score = q_value + exploration
        if score > best_score:
            best_score = score
            best_child = child
    assert best_child is not None
    return best_child


def _run_simulation(
    root: SearchNode,
    model: PolicyValueNet,
    engine: RustEngine,
    config: MctsConfig,
    device: torch.device | str,
    input_view: InputView,
    history: EncodedHistory | None,
    history_plies: int,
) -> None:
    node = root
    path = [node]
    edge_discounts: List[float] = []
    depth = 0
    while node.children and depth < config.max_depth:
        child = _select_child(node, config)
        if child.node is None:
            child_state = _apply_search_action(node.state, child.action, engine)
            child_player = child_state["currentPlayer"]
            child_tune_count = node.tune_count + 1 if child_player == node.player and _is_tune(child.action) else 0
            child.node = SearchNode(
                state=child_state,
                player=child_player,
                tune_count=child_tune_count,
                prior=child.prior,
            )
        node = child.node
        path.append(node)
        edge_discounts.append(1.0 if _is_tune(child.action) else config.value_discount)
        depth += 1

    if node.state["status"] != "playing":
        value = result_value(node.state["status"], node.player)
    elif node.evaluation is not None:
        value = node.evaluation
    else:
        value = _expand(node, model, engine, config, device, input_view, history, history_plies)

    for index in range(len(path) - 1, -1, -1):
        current = path[index]
        current.visits += 1
        current.value_sum += value
        if index > 0:
            value *= edge_discounts[index - 1]
            if path[index - 1].player != current.player:
                value = -value


def select_mcts_action(
    model: PolicyValueNet,
    state: Dict[str, Any],
    engine: RustEngine,
    config: MctsConfig | None = None,
    device: torch.device | str = "cpu",
    input_view: InputView = "base",
    history: EncodedHistory | None = None,
    history_plies: int = 1,
    tune_count: int = 0,
    rng: np.random.Generator | None = None,
) -> MctsResult:
    model.eval()
    config = config or MctsConfig()
    if not 0 < config.value_discount <= 1:
        raise ValueError("MCTS value discount must be in (0, 1].")
    state = without_undo_history(state)
    root = SearchNode(state=state, player=state["currentPlayer"], tune_count=tune_count)
    root_value = _expand(root, model, engine, config, device, input_view, history, history_plies, root_noise=True, rng=rng)
    if not root.children:
        raise ValueError("MCTS requires at least one legal search action")

    for _ in range(max(0, config.simulations)):
        _run_simulation(root, model, engine, config, device, input_view, history, history_plies)

    children = sorted(root.children.values(), key=lambda child: child.node.visits if child.node is not None else 0, reverse=True)
    best = children[0]
    assert best.node is not None
    policy = [
        {
            "action": child.action,
            "prior": child.prior,
            "visits": child.node.visits if child.node is not None else 0,
            "value": _child_value_for_parent(root, child.node, 1.0 if _is_tune(child.action) else config.value_discount) if child.node is not None else 0.0,
        }
        for child in children
    ]
    return MctsResult(
        action=best.action,
        visits=best.node.visits,
        value=_child_value_for_parent(root, best.node, 1.0 if _is_tune(best.action) else config.value_discount),
        root_value=root_value,
        policy=policy,
    )


def _force_tune_result(result: MctsResult) -> MctsResult:
    tune_rows = [row for row in result.policy if _is_tune(row["action"])]
    if not tune_rows:
        return result
    best = max(tune_rows, key=lambda row: (row["visits"], row["prior"], row["value"]))
    return MctsResult(
        action=best["action"],
        visits=int(best["visits"]),
        value=float(best["value"]),
        root_value=result.root_value,
        policy=tune_rows,
    )


def _sample_visit_result(result: MctsResult, rng: np.random.Generator) -> MctsResult:
    weights = np.asarray([max(0, row["visits"]) for row in result.policy], dtype=np.float64)
    if weights.sum() <= 0:
        return result
    row = result.policy[int(rng.choice(len(result.policy), p=weights / weights.sum()))]
    return MctsResult(
        action=row["action"],
        visits=int(row["visits"]),
        value=float(row["value"]),
        root_value=result.root_value,
        policy=result.policy,
    )


def sample_from_mcts_result(
    state: Dict[str, Any],
    engine: RustEngine,
    result: MctsResult,
    input_view: InputView = "base",
    history: EncodedHistory | None = None,
    history_plies: int = 1,
) -> Sample:
    move_actions = engine.legal_actions(state)
    tune_actions = legal_tuning_actions(state)
    board, side, move_mask = encode_state(state, engine, move_actions, input_view=input_view)
    tune_mask = legal_tuning_mask(tune_actions)
    kind_policy = np.zeros((2,), dtype=np.float32)
    move_policy = np.zeros_like(move_mask, dtype=np.float32)
    tuning_policy = np.zeros_like(tune_mask, dtype=np.float32)
    total_visits = sum(max(0, row["visits"]) for row in result.policy)
    if total_visits <= 0:
        total_visits = len(result.policy)
        weights = [1.0 for _row in result.policy]
    else:
        weights = [float(row["visits"]) for row in result.policy]

    for row, weight in zip(result.policy, weights):
        probability = float(weight) / float(total_visits)
        action = row["action"]
        if _is_tune(action):
            kind_policy[1] += probability
            tuning_policy[tuning_action_index(action)] += probability
        else:
            kind_policy[0] += probability
            move_policy[action_index(action)] += probability

    selected = result.action
    sample = Sample(
        board=board,
        side=side,
        legal_mask=move_mask,
        action_index=action_index(selected) if not _is_tune(selected) else -100,
        player=state["currentPlayer"],
        action_kind=1 if _is_tune(selected) else 0,
        legal_tuning_mask=tune_mask,
        tuning_action_index=tuning_action_index(selected) if _is_tune(selected) else -100,
        kind_policy=kind_policy,
        move_policy=move_policy if move_policy.sum() > 0 else None,
        tuning_policy=tuning_policy if tuning_policy.sum() > 0 else None,
        metadata={
            "source": "mcts_model",
            "legal_count": len(move_actions),
            "legal_tuning_count": int(tune_mask.sum()),
            "mcts_visits": int(total_visits),
            "mcts_root_value": result.root_value,
            "mcts_selected_value": result.value,
        },
    )
    history_arrays = _history_arrays(board, side, history, history_plies)
    if history_arrays is not None:
        sample.history_board, sample.history_side = history_arrays
        sample.metadata["history_plies"] = history_plies
    return sample


def _discount_mcts_values(samples: List[Sample], discount: float) -> None:
    """Discount each state's outcome by the actual piece moves still to come."""
    if not 0 < discount <= 1:
        raise ValueError("MCTS value discount must be in (0, 1].")
    remaining_plies = 0
    for sample in reversed(samples):
        if sample.action_kind == 0:
            remaining_plies += 1
        sample.metadata.update(outcome_value=sample.value, remaining_plies=remaining_plies,
                               value_discount=discount)
        sample.value *= discount ** remaining_plies


def mcts_selfplay_records(
    engine: RustEngine,
    model: PolicyValueNet,
    games: int,
    max_plies: int,
    seed: int,
    config: MctsConfig,
    device: torch.device | str = "cpu",
    input_view: InputView = "base",
    history_plies: int = 1,
    cap_value: CapValueMode = "material",
    initial_states: Sequence[Dict[str, Any]] | None = None,
    game_offset: int = 0,
    on_game_complete: Callable[[int, GameRecord], None] | None = None,
    on_position: Callable[..., None] | None = None,
) -> List[GameRecord]:
    records: List[GameRecord] = []
    model.eval()
    for game_index in range(games):
        rng = np.random.default_rng(seed + game_offset + game_index)
        state = (
            initial_states[game_index]
            if initial_states is not None and game_index < len(initial_states)
            else None
        )
        if state is None:
            from .engine import load_initial_state
            state = load_initial_state()
        state = without_undo_history(state)
        if on_position is not None:
            on_position(state, 0)
        samples: List[Sample] = []
        stats = GameStats()
        repetitions = {repetition_key(state): 1} if config.repetition_limit > 0 else {}
        history: List[Tuple[np.ndarray, np.ndarray]] = []
        for ply in range(max_plies):
            if state["status"] != "playing":
                break
            before_state = state
            player = state["currentPlayer"]
            tune_count = 0
            turn_tunes = 0
            effective_before = state
            while state["status"] == "playing" and state["currentPlayer"] == player:
                if not engine.legal_actions(state) and not tuning_limit_allows(tune_count, config.max_tuning_actions):
                    state = _no_move_loss(state)
                    break
                result = select_mcts_action(
                    model,
                    state,
                    engine,
                    config=config,
                    device=device,
                    input_view=input_view,
                    history=history,
                    history_plies=history_plies,
                    tune_count=tune_count,
                    rng=rng,
                )
                tune_actions = legal_tuning_actions(state) if tuning_limit_allows(tune_count, config.max_tuning_actions) else []
                force_tune = False
                if tune_actions:
                    if tune_count == 0 and config.force_first_tune_prob > 0:
                        force_tune = rng.random() < config.force_first_tune_prob
                    if not force_tune and config.force_tune_prob > 0:
                        force_tune = rng.random() < config.force_tune_prob
                if force_tune:
                    result = _force_tune_result(result)
                if ply < config.sample_first_plies:
                    result = _sample_visit_result(result, rng)
                samples.append(
                    sample_from_mcts_result(
                        state,
                        engine,
                        result,
                        input_view=input_view,
                        history=history,
                        history_plies=history_plies,
                    )
                )
                if _is_tune(result.action):
                    state = without_undo_history(engine.apply_tuning(state, result.action))  # type: ignore[arg-type]
                    tune_count += 1
                    turn_tunes += 1
                    if on_position is not None:
                        on_position(state, ply, result.action)
                else:
                    state = _apply_search_action(state, result.action, engine)
                    if on_position is not None:
                        on_position(state, ply + 1, result.action)
                    break
            stats.ai_turns_by_player[player] += 1
            record_turn_tactics(stats, before_state, state)
            stats.tune_actions_by_player[player] += turn_tunes
            if turn_tunes > 0:
                stats.tune_turns_by_player[player] += 1
            stats.effective_tune_changes_by_player[player] += sum(
                1
                for piece_type in effective_before["components"][player]
                for before, after in zip(effective_before["components"][player][piece_type], state["components"][player][piece_type])
                if before != after
            )
            if history_plies > 1:
                moves = engine.legal_actions(before_state)
                board, side, _mask = encode_state(before_state, engine, moves, input_view=input_view)
                history.append((board, side))
                history = history[-(history_plies - 1):]
            stats.plies = ply + 1
            if config.repetition_limit > 0 and state['status'] == 'playing':
                key = repetition_key(state)
                repetitions[key] = repetitions.get(key, 0) + 1
                if repetitions[key] >= config.repetition_limit:
                    state = {**state, 'terminationReason': 'threefold_repetition',
                             'message': 'Draw by threefold repetition'}
                    stats.termination_reason = 'threefold_repetition'
                    stats.plies_avoided_by_repetition = max_plies - stats.plies
                    break
        if on_position is not None:
            on_position(state, stats.plies, complete=True)
        _assign_values(samples, state, cap_value)
        _discount_mcts_values(samples, config.value_discount)
        stats.status = state["status"]
        winner = None
        if state["status"] == "red-won":
            winner = "red"
        elif state["status"] == "blue-won":
            winner = "blue"
        stats.winner = winner
        stats.final_piece_counts = _piece_counts(state)
        records.append(GameRecord(samples=samples, stats=stats, final_state=state))
        if on_game_complete is not None:
            on_game_complete(game_offset + game_index, records[-1])
    return records


_worker_model: PolicyValueNet | None = None
_worker_engine: RustEngine | None = None


def _init_mcts_worker(model: PolicyValueNet | None, inference_spec: Any = None) -> None:
    global _worker_model, _worker_engine
    torch.set_num_threads(1)
    if inference_spec is not None:
        from .shared_inference import SharedInferenceClient
        _worker_model = SharedInferenceClient(inference_spec)
    else:
        assert model is not None
        _worker_model = model
    _worker_model.eval()
    _worker_engine = RustEngine(retain_history=False)


def _mcts_worker_game(
    task: Tuple[int, int, int, MctsConfig, InputView, int, CapValueMode, Dict[str, Any] | None],
) -> GameRecord:
    assert _worker_model is not None and _worker_engine is not None
    game_index, max_plies, seed, config, input_view, history_plies, cap_value, initial_state = task
    from .spectator import SpectatorFeed
    from .mcts_settings import worker_config
    config = worker_config(config, seed)
    with SpectatorFeed(game_index, seed, config.simulations, max_plies, config.max_tuning_actions, config.value_discount) as feed:
        return mcts_selfplay_records(
            _worker_engine,
            _worker_model,
            games=1,
            max_plies=max_plies,
            seed=seed,
            config=config,
            device="cpu",
            input_view=input_view,
            history_plies=history_plies,
            cap_value=cap_value,
            initial_states=[initial_state] if initial_state is not None else None,
            game_offset=game_index,
            on_position=feed.publish,
        )[0]


def parallel_mcts_selfplay_records(
    model: PolicyValueNet,
    games: int,
    workers: int,
    max_plies: int,
    seed: int,
    config: MctsConfig,
    input_view: InputView = "base",
    history_plies: int = 1,
    cap_value: CapValueMode = "material",
    initial_states: Sequence[Dict[str, Any]] | None = None,
    on_game_complete: Callable[[int, GameRecord], None] | None = None,
    inference_device: str | None = None,
    inference_max_batch: int = 32,
    inference_wait_ms: float = 2.0,
    inference_stats: Dict[str, Any] | None = None,
) -> List[GameRecord]:
    if games < 0:
        raise ValueError("MCTS games must be nonnegative")
    if games == 0:
        return []
    if workers < 1:
        raise ValueError("MCTS workers must be at least 1")
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("Parallel MCTS self-play currently requires a CPU model")
    if (workers == 1 or games <= 1) and inference_device is None:
        engine = RustEngine(retain_history=False)
        try:
            return mcts_selfplay_records(
                engine, model, games, max_plies, seed, config,
                device="cpu", input_view=input_view, history_plies=history_plies,
                cap_value=cap_value, initial_states=initial_states,
                on_game_complete=on_game_complete,
            )
        finally:
            engine.close()
    tasks = [
        (
            game_index, max_plies, seed, config, input_view, history_plies, cap_value,
            initial_states[game_index] if initial_states is not None and game_index < len(initial_states) else None,
        )
        for game_index in range(games)
    ]
    if inference_device is not None:
        from .shared_inference import SharedInferencePool
        service_context = SharedInferencePool(model, min(workers, games), inference_device,
                                              inference_max_batch, inference_wait_ms)
    else:
        service_context = nullcontext(None)
    with service_context as service, ProcessPoolExecutor(
        max_workers=min(workers, games),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_mcts_worker,
        initargs=(None, service.spec) if service is not None else (model,),
    ) as executor:
        futures = {executor.submit(_mcts_worker_game, task): task[0] for task in tasks}
        completed: Dict[int, GameRecord] = {}
        for future in as_completed(futures):
            game_index = futures[future]
            completed[game_index] = future.result()
            if on_game_complete is not None:
                on_game_complete(game_index, completed[game_index])
    if inference_stats is not None and service is not None:
        inference_stats.update(service.stats)
    return [completed[index] for index in range(games)]
