from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Literal

from .engine import RustEngine, load_initial_state
from .eval import aggregate, load_model
from .selfplay import (
    GameRecord,
    GameStats,
    select_model_action,
    select_model_full_turn,
)
from .train import resolve_device


Policy = Literal["model", "heuristic", "easy", "random"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run head-to-head policy matches.")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--max-plies", type=int, default=150)
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--red", choices=("model", "heuristic", "easy", "random"), default="model")
    parser.add_argument("--blue", choices=("model", "heuristic", "easy", "random"), default="heuristic")
    parser.add_argument("--checkpoint", type=Path, default=Path("training/checkpoints/rust_batch_2000x150_policy_value.pt"))
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--model-arch", choices=("conv", "residual", "transformer", "sequence_transformer"), default=None)
    parser.add_argument("--input-view", choices=("base", "piece_identity"), default=None)
    parser.add_argument("--history-plies", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--full-policy", action="store_true", help="Let model turns choose tuning actions before moving.")
    parser.add_argument("--max-tuning-actions", type=int, default=3)
    parser.add_argument("--no-pressure", action="store_true", help="Skip exact pressure and rescue metrics for faster runs.")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _piece_map(state: Dict) -> Dict:
    return {piece["id"]: piece for piece in state["pieces"]}


def _winner(status: str) -> str | None:
    if status == "red-won":
        return "red"
    if status == "blue-won":
        return "blue"
    return None


def _unstable_ids_for_player(state: Dict[str, Any], player: str) -> set[str]:
    return {
        piece["id"]
        for piece in state["pieces"]
        if piece["owner"] == player and piece.get("unstable", False)
    }


def _piece_counts(state: Dict[str, Any]) -> Dict[str, int]:
    counts = {"red": 0, "blue": 0}
    for piece in state["pieces"]:
        counts[piece["owner"]] += 1
    return counts


def _component_change_count(before: Dict[str, Any], after: Dict[str, Any], player: str) -> int:
    before_components = before["components"][player]
    after_components = after["components"][player]
    return sum(
        1
        for piece_type in before_components
        for left, right in zip(before_components[piece_type], after_components[piece_type])
        if left != right
    )


def _no_move_loss(state: Dict[str, Any]) -> Dict[str, Any]:
    winner = "red" if state["currentPlayer"] == "blue" else "blue"
    return {
        **state,
        "status": f"{winner}-won",
        "selectedPieceId": None,
        "message": f"{state['currentPlayer'].capitalize()} has no legal move",
    }


def _record_losses(stats: GameStats, before: Dict[str, Any], after: Dict[str, Any]) -> list[str]:
    after_pieces = _piece_map(after)
    lost_ids = [piece_id for piece_id in before if piece_id not in after_pieces]
    for piece_id in lost_ids:
        piece = before[piece_id]
        owner = piece["owner"]
        piece_type = piece["type"]
        stats.losses_by_player[owner] += 1
        stats.losses_by_piece_type[piece_type] = stats.losses_by_piece_type.get(piece_type, 0) + 1
        if stats.first_loss_player is None:
            stats.first_loss_player = owner
            stats.first_loss_piece_type = piece_type
    return lost_ids


def play_match_game(
    engine: RustEngine,
    policies: Dict[str, Policy],
    model,
    device,
    max_plies: int,
    seed: int,
    temperature: float,
    input_view: str,
    full_policy: bool = False,
    max_tuning_actions: int = 3,
    collect_metrics: bool = True,
) -> GameRecord:
    state = load_initial_state()
    stats = GameStats()
    encoded_history = []
    if model is not None:
        model.eval()

    for ply in range(max_plies):
        if state["status"] != "playing":
            break

        player = state["currentPlayer"]
        policy = policies[player]
        before_state = state
        before = _piece_map(state)
        before_unstable = _unstable_ids_for_player(state, player)
        before_counts = _piece_counts(state)

        if collect_metrics:
            for side in ("red", "blue"):
                stats.pressure_sum[side] += len(engine.player_actions(state, side))
            stats.pressure_samples += 1
            if before_unstable:
                stats.rescue_opportunities += 1

        if policy == "model":
            if model is None:
                raise ValueError("model policy requires --checkpoint")
            actions = engine.legal_actions(state)
            if full_policy:
                next_state, _samples, tune_actions = select_model_full_turn(
                    model,
                    state,
                    engine,
                    temperature=temperature,
                    device=device,
                    record_samples=False,
                    input_view=input_view,
                    max_tuning_actions=max_tuning_actions,
                    history=encoded_history,
                    history_plies=getattr(model, "history_plies", 1),
                )
                if next_state["status"] == "playing" and not engine.legal_actions(next_state):
                    next_state = _no_move_loss(next_state)
                stats.ai_turns_by_player[player] += 1
                stats.tune_actions_by_player[player] += tune_actions
                stats.effective_tune_changes_by_player[player] += _component_change_count(state, next_state, player)
                if tune_actions > 0:
                    stats.tune_turns_by_player[player] += 1
                state = next_state
            else:
                if not actions:
                    state = _no_move_loss(state)
                else:
                    action, _sample = select_model_action(
                        model,
                        state,
                        engine,
                        actions,
                        temperature=temperature,
                        device=device,
                        record_sample=False,
                        input_view=input_view,
                        history=encoded_history,
                        history_plies=getattr(model, "history_plies", 1),
                    )
                    state = engine.apply_action(state, action, analyze_checkmate=False)
                stats.ai_turns_by_player[player] += 1
        elif policy == "heuristic":
            state = engine.play_heuristic_turn(
                state,
                player=player,
                seed=seed + ply,
                variety=0.55,
                time_budget_ms=10,
            )
            tune_changes = _component_change_count(before_state, state, player)
            stats.ai_turns_by_player[player] += 1
            stats.tune_actions_by_player[player] += tune_changes
            stats.effective_tune_changes_by_player[player] += tune_changes
            if tune_changes > 0:
                stats.tune_turns_by_player[player] += 1
        elif policy == "easy":
            state = engine.play_easy_turn(
                state,
                player=player,
                seed=seed + ply,
                variety=0.0,
                time_budget_ms=10,
            )
            stats.ai_turns_by_player[player] += 1
        else:
            actions = engine.legal_actions(state)
            if not actions:
                state = _no_move_loss(state)
            else:
                action = actions[(seed + ply) % len(actions)]
                state = engine.apply_action(state, action, analyze_checkmate=False)
            stats.ai_turns_by_player[player] += 1

        lost_ids = _record_losses(stats, before, state)
        if collect_metrics and before_unstable:
            after_unstable = _unstable_ids_for_player(state, player)
            if before_unstable.isdisjoint(set(lost_ids)) and before_unstable.isdisjoint(after_unstable):
                stats.rescues += 1

        winner = _winner(state["status"])
        if winner is not None:
            loser = "blue" if winner == "red" else "red"
            after_counts = _piece_counts(state)
            stats.min_winner_pieces = min(before_counts[winner], after_counts[winner])
            stats.max_loser_pieces = max(before_counts[loser], after_counts[loser])

        stats.plies = ply + 1
        if getattr(model, "history_plies", 1) > 1:
            from .encoding import encode_state
            actions_for_history = engine.legal_actions(before_state)
            board, side, _mask = encode_state(before_state, engine, actions_for_history, input_view=input_view)
            encoded_history.append((board, side))
            encoded_history = encoded_history[-(getattr(model, "history_plies", 1) - 1):]

    stats.status = state["status"]
    stats.winner = _winner(state["status"])
    stats.final_piece_counts = _piece_counts(state)
    return GameRecord(samples=[], stats=stats, final_state=state)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    engine = RustEngine()
    if "model" in (args.red, args.blue):
        model, input_view = load_model(
            args.checkpoint,
            args.hidden_size,
            device,
            model_arch=args.model_arch,
            input_view=args.input_view,
            history_plies=args.history_plies,
        )
    else:
        model = None
        input_view = args.input_view or "base"
    policies: Dict[str, Policy] = {"red": args.red, "blue": args.blue}
    records = [
        play_match_game(
            engine,
            policies=policies,
            model=model,
            device=device,
            max_plies=args.max_plies,
            seed=args.seed + game,
            temperature=args.temperature,
            input_view=input_view,
            full_policy=args.full_policy,
            max_tuning_actions=args.max_tuning_actions,
            collect_metrics=not args.no_pressure,
        )
        for game in range(args.games)
    ]
    summary = {
        "red": args.red,
        "blue": args.blue,
        "checkpoint": str(args.checkpoint) if "model" in (args.red, args.blue) else None,
        "max_plies": args.max_plies,
        "temperature": args.temperature,
        **aggregate(records),
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(summary)


if __name__ == "__main__":
    main()
