from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch

from .encoding import (
    InputView,
    action_index,
    encode_state,
    legal_tuning_actions,
    legal_tuning_mask,
    tuning_action_index,
)
from .engine import Action, RustEngine, TuningAction
from .model import PolicyValueNet, masked_policy_logits
from .selfplay import EncodedHistory, _history_arrays, result_value
from .selfplay import CapValueMode, GameRecord, GameStats, Sample, _assign_values, _piece_counts


SearchAction = Union[Action, TuningAction]


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
    simulations: int = 32
    c_puct: float = 1.5
    top_k: int = 16
    max_depth: int = 16
    max_tuning_actions: int = 3
    policy_temperature: float = 1.0
    dirichlet_alpha: float = 0.0
    exploration_fraction: float = 0.0


@dataclass
class SearchChild:
    action: SearchAction
    prior: float
    node: SearchNode


@dataclass
class SearchNode:
    state: Dict[str, Any]
    player: str
    tune_count: int
    prior: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
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
        kind_logits, move_logits, tune_logits = model.full_policy(
            board_tensor,
            side_tensor,
            history_board=history_board_tensor,
            history_side=history_side_tensor,
        )
        _policy_logits, values = model(
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
    tunes = legal_tuning_actions(state) if tune_count < max_tuning_actions else []
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
    scored: List[Tuple[float, SearchAction]] = []
    for action in tune_actions:
        scored.append((float(tune_priors[tuning_action_index(action)]), action))
    for action in move_actions:
        scored.append((float(move_priors[action_index(action)]), action))
    scored.sort(key=lambda item: item[0], reverse=True)
    scored = scored[:max(1, config.top_k)]

    total_prior = sum(max(0.0, prior) for prior, _action in scored)
    if total_prior <= 0:
        normalized = [(1.0 / len(scored), action) for _prior, action in scored]
    else:
        normalized = [(max(0.0, prior) / total_prior, action) for prior, action in scored]

    if root_noise and config.dirichlet_alpha > 0 and config.exploration_fraction > 0 and len(normalized) > 1:
        noise = np.random.default_rng().dirichlet([config.dirichlet_alpha] * len(normalized))
        normalized = [
            (
                prior * (1.0 - config.exploration_fraction)
                + float(noise[index]) * config.exploration_fraction,
                action,
            )
            for index, (prior, action) in enumerate(normalized)
        ]

    for prior, action in normalized:
        child_state = _apply_search_action(node.state, action, engine)
        child_player = child_state["currentPlayer"]
        child_tune_count = node.tune_count + 1 if child_player == node.player and _is_tune(action) else 0
        child = SearchNode(
            state=child_state,
            player=child_player,
            tune_count=child_tune_count,
            prior=prior,
        )
        node.children[_action_key(action)] = SearchChild(action=action, prior=prior, node=child)
    return value


def _apply_search_action(state: Dict[str, Any], action: SearchAction, engine: RustEngine) -> Dict[str, Any]:
    if _is_tune(action):
        return engine.apply_tuning(state, action)  # type: ignore[arg-type]
    next_state = engine.apply_action(state, action, analyze_checkmate=False)  # type: ignore[arg-type]
    if next_state["status"] == "playing" and not engine.legal_actions(next_state):
        return _no_move_loss(next_state)
    return next_state


def _child_value_for_parent(parent: SearchNode, child: SearchNode) -> float:
    return child.value if child.player == parent.player else -child.value


def _select_child(node: SearchNode, config: MctsConfig) -> SearchChild:
    parent_visits = max(1, node.visits)
    best_score = -float("inf")
    best_child: SearchChild | None = None
    for child in node.children.values():
        q_value = _child_value_for_parent(node, child.node)
        exploration = config.c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.node.visits)
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
    depth = 0
    while node.children and depth < config.max_depth:
        child = _select_child(node, config)
        node = child.node
        path.append(node)
        depth += 1

    if node.state["status"] != "playing":
        value = result_value(node.state["status"], node.player)
    else:
        value = _expand(node, model, engine, config, device, input_view, history, history_plies)

    for index in range(len(path) - 1, -1, -1):
        current = path[index]
        current.visits += 1
        current.value_sum += value
        if index > 0 and path[index - 1].player != current.player:
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
) -> MctsResult:
    model.eval()
    config = config or MctsConfig()
    root = SearchNode(state=state, player=state["currentPlayer"], tune_count=tune_count)
    root_value = _expand(root, model, engine, config, device, input_view, history, history_plies, root_noise=True)
    if not root.children:
        raise ValueError("MCTS requires at least one legal search action")

    for _ in range(max(0, config.simulations)):
        _run_simulation(root, model, engine, config, device, input_view, history, history_plies)

    children = sorted(root.children.values(), key=lambda child: child.node.visits, reverse=True)
    best = children[0]
    policy = [
        {
            "action": child.action,
            "prior": child.prior,
            "visits": child.node.visits,
            "value": _child_value_for_parent(root, child.node),
        }
        for child in children
    ]
    return MctsResult(
        action=best.action,
        visits=best.node.visits,
        value=_child_value_for_parent(root, best.node),
        root_value=root_value,
        policy=policy,
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
) -> List[GameRecord]:
    records: List[GameRecord] = []
    model.eval()
    for game_index in range(games):
        state = (
            initial_states[game_index]
            if initial_states is not None and game_index < len(initial_states)
            else None
        )
        if state is None:
            from .engine import load_initial_state
            state = load_initial_state()
        samples: List[Sample] = []
        stats = GameStats()
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
                if not engine.legal_actions(state) and tune_count >= config.max_tuning_actions:
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
                )
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
                    state = engine.apply_tuning(state, result.action)  # type: ignore[arg-type]
                    tune_count += 1
                    turn_tunes += 1
                else:
                    state = _apply_search_action(state, result.action, engine)
                    break
            stats.ai_turns_by_player[player] += 1
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
        _assign_values(samples, state, cap_value)
        stats.status = state["status"]
        winner = None
        if state["status"] == "red-won":
            winner = "red"
        elif state["status"] == "blue-won":
            winner = "blue"
        stats.winner = winner
        stats.final_piece_counts = _piece_counts(state)
        records.append(GameRecord(samples=samples, stats=stats, final_state=state))
    return records
