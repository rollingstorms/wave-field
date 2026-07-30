from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Literal

from .engine import RustEngine, load_initial_state
from .eval import aggregate, load_model
from .selfplay import GameRecord, GameStats, select_model_action
from .train import resolve_device


Policy = Literal["model", "heuristic", "random"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run head-to-head policy matches.")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--max-plies", type=int, default=150)
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--red", choices=("model", "heuristic", "random"), default="model")
    parser.add_argument("--blue", choices=("model", "heuristic", "random"), default="heuristic")
    parser.add_argument("--checkpoint", type=Path, default=Path("training/checkpoints/rust_batch_2000x150_policy_value.pt"))
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _piece_map(state: Dict) -> Dict:
    return {piece["id"]: piece for piece in state["pieces"]}


def _winner(status: str) -> str | None:
    if status == "red-won":
        return "red"
    if status == "blue-won":
        return "blue"
    return None


def play_match_game(
    engine: RustEngine,
    policies: Dict[str, Policy],
    model,
    device,
    max_plies: int,
    seed: int,
    temperature: float,
    input_view: str,
) -> GameRecord:
    state = load_initial_state()
    stats = GameStats()

    for ply in range(max_plies):
        if state["status"] != "playing":
            break

        player = state["currentPlayer"]
        policy = policies[player]
        before = _piece_map(state)

        if policy == "model":
            actions = engine.legal_actions(state)
            if not actions:
                break
            action, _sample = select_model_action(
                model,
                state,
                engine,
                actions,
                temperature=temperature,
                device=device,
                record_sample=False,
                input_view=input_view,
            )
            state = engine.apply_action(state, action, analyze_checkmate=False)
        elif policy == "heuristic":
            state = engine.play_heuristic_turn(
                state,
                player=player,
                seed=seed + ply,
                variety=0.55,
                time_budget_ms=10,
            )
        else:
            actions = engine.legal_actions(state)
            if not actions:
                break
            action = actions[(seed + ply) % len(actions)]
            state = engine.apply_action(state, action, analyze_checkmate=False)

        after = _piece_map(state)
        for piece_id, piece in before.items():
            if piece_id in after:
                continue
            owner = piece["owner"]
            piece_type = piece["type"]
            stats.losses_by_player[owner] += 1
            stats.losses_by_piece_type[piece_type] = stats.losses_by_piece_type.get(piece_type, 0) + 1
            if stats.first_loss_player is None:
                stats.first_loss_player = owner
                stats.first_loss_piece_type = piece_type

        stats.plies = ply + 1

    stats.status = state["status"]
    stats.winner = _winner(state["status"])
    counts = {"red": 0, "blue": 0}
    for piece in state["pieces"]:
        counts[piece["owner"]] += 1
    stats.final_piece_counts = counts
    return GameRecord(samples=[], stats=stats, final_state=state)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    engine = RustEngine()
    model, input_view = load_model(args.checkpoint, args.hidden_size, device)
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
        )
        for game in range(args.games)
    ]
    print({"red": args.red, "blue": args.blue, **aggregate(records)})


if __name__ == "__main__":
    main()
