from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch

from .encoding import SIDE_SIZE, board_channels_for_view
from .engine import RustEngine
from .model import PolicyValueNet
from .selfplay import PolicyMode, selfplay_records, session_model_selfplay_records
from .train import resolve_device


PLAYERS = ("red", "blue")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Wave Field self-play policies.")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--policy", choices=("random", "model", "heuristic", "easy"), default="random")
    parser.add_argument("--checkpoint", type=Path, default=Path("training/checkpoints/policy_value.pt"))
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument(
        "--model-arch",
        choices=("conv", "residual", "transformer", "sequence_transformer", "encoder_sequence"),
        default=None,
    )
    parser.add_argument("--input-view", choices=("base", "piece_identity"), default=None)
    parser.add_argument("--history-plies", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--session", action="store_true", help="Use fast Rust session rollout for model eval.")
    parser.add_argument("--no-pressure", action="store_true", help="Skip exact pressure during session eval for faster bulk runs.")
    parser.add_argument("--full-policy", action="store_true", help="Let model choose tuning actions before moving.")
    parser.add_argument("--max-tuning-actions", type=int, default=3)
    parser.add_argument("--rollout-batch-size", type=int, default=128)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def load_model(
    path: Path,
    hidden_size: int,
    device: torch.device,
    model_arch: str | None = None,
    input_view: str | None = None,
    history_plies: int | None = None,
) -> tuple[PolicyValueNet, str]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    resolved_arch = model_arch or checkpoint.get("model_arch", "conv")
    resolved_view = input_view or checkpoint.get("input_view", "base")
    resolved_hidden = int(checkpoint.get("hidden_size", hidden_size))
    resolved_history = int(history_plies or checkpoint.get("history_plies", 1))
    resolved_encoder_arch = checkpoint.get("encoder_arch", "transformer")
    resolved_encoder_hidden = int(checkpoint.get("encoder_hidden_size", resolved_hidden))
    resolved_freeze_encoder = bool(checkpoint.get("freeze_encoder", True))
    model = PolicyValueNet(
        hidden_size=resolved_hidden,
        board_channels=board_channels_for_view(resolved_view),
        side_size=SIDE_SIZE,
        architecture=resolved_arch,
        history_plies=resolved_history,
        encoder_arch=resolved_encoder_arch,
        encoder_hidden_size=resolved_encoder_hidden,
        freeze_encoder=resolved_freeze_encoder,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    return model, resolved_view


def aggregate(records: List[Any]) -> Dict[str, Any]:
    winners = Counter(record.stats.winner or "capped" for record in records)
    first_loss_winner = Counter()
    first_loss_loser = Counter()
    losses_by_piece_type = Counter()
    losses_by_player = Counter()
    wins_by_piece_count: Dict[str, Counter[int]] = defaultdict(Counter)
    underdog_wins = Counter()
    pressure_totals = Counter()
    ai_turns = Counter()
    tune_turns = Counter()
    tune_actions = Counter()
    effective_tune_changes = Counter()
    rescue_opportunities = 0
    rescues = 0
    plies = [record.stats.plies for record in records]
    final_margins = []

    for record in records:
        stats = record.stats
        winner = stats.winner
        if winner and stats.first_loss_player:
            if stats.first_loss_player == winner:
                first_loss_winner[winner] += 1
            else:
                first_loss_loser[stats.first_loss_player] += 1
        losses_by_piece_type.update(stats.losses_by_piece_type)
        losses_by_player.update(stats.losses_by_player)
        rescue_opportunities += stats.rescue_opportunities
        rescues += stats.rescues
        pressure_totals.update(stats.pressure_sum)
        ai_turns.update(getattr(stats, "ai_turns_by_player", {}))
        tune_turns.update(getattr(stats, "tune_turns_by_player", {}))
        tune_actions.update(getattr(stats, "tune_actions_by_player", {}))
        effective_tune_changes.update(getattr(stats, "effective_tune_changes_by_player", {}))

        if winner:
            winner_count = stats.final_piece_counts[winner]
            loser = "blue" if winner == "red" else "red"
            loser_count = stats.final_piece_counts[loser]
            wins_by_piece_count[winner][winner_count] += 1
            if winner_count < loser_count:
                underdog_wins[winner] += 1
        final_margins.append(stats.final_piece_counts["red"] - stats.final_piece_counts["blue"])

    pressure_samples = sum(record.stats.pressure_samples for record in records)
    decisive = sum(1 for record in records if record.stats.decisive)
    return {
        "games": len(records),
        "decisive": decisive,
        "decisive_rate": decisive / len(records) if records else 0.0,
        "capped": len(records) - decisive,
        "capped_rate": (len(records) - decisive) / len(records) if records else 0.0,
        "mean_plies": sum(plies) / len(plies) if plies else 0.0,
        "ply_distribution": numeric_distribution(plies),
        "wins": dict(winners),
        "win_rates": {
            player: winners[player] / len(records) if records else 0.0
            for player in PLAYERS
        },
        "first_loss_team_won": dict(first_loss_winner),
        "first_loss_team_lost": dict(first_loss_loser),
        "losses_by_player": dict(losses_by_player),
        "losses_by_piece_type": dict(losses_by_piece_type),
        "piece_loss_frequency_per_game": {
            piece_type: count / len(records) for piece_type, count in losses_by_piece_type.items()
        } if records else {},
        "rescue_opportunities": rescue_opportunities,
        "rescues": rescues,
        "rescue_rate": rescues / rescue_opportunities if rescue_opportunities else 0.0,
        "avg_pressure": {
            player: pressure_totals[player] / pressure_samples if pressure_samples else 0.0
            for player in PLAYERS
        },
        "tuning": {
            player: {
                "ai_turns": ai_turns[player],
                "tune_turns": tune_turns[player],
                "tune_turn_rate": tune_turns[player] / ai_turns[player] if ai_turns[player] else 0.0,
                "tune_actions": tune_actions[player],
                "tune_actions_per_turn": tune_actions[player] / ai_turns[player] if ai_turns[player] else 0.0,
                "effective_changes": effective_tune_changes[player],
                "effective_changes_per_turn": effective_tune_changes[player] / ai_turns[player] if ai_turns[player] else 0.0,
            }
            for player in PLAYERS
        },
        "avg_final_material_balance_red": (
            sum(final_margins) / len(final_margins) if final_margins else 0.0
        ),
        "underdog_wins": dict(underdog_wins),
        "wins_by_final_piece_count": {
            player: dict(counts) for player, counts in wins_by_piece_count.items()
        },
    }


def numeric_distribution(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
        return float(ordered[index])

    return {
        "min": float(ordered[0]),
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "max": float(ordered[-1]),
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print(f"games={summary['games']} decisive={summary['decisive']} capped={summary['capped']}")
    print(
        f"mean_plies={summary['mean_plies']:.2f} "
        f"ply_distribution={summary['ply_distribution']} "
        f"wins={summary['wins']} win_rates={summary['win_rates']}"
    )
    print(f"first_loss_team_won={summary['first_loss_team_won']}")
    print(f"first_loss_team_lost={summary['first_loss_team_lost']}")
    print(f"losses_by_player={summary['losses_by_player']}")
    print(f"losses_by_piece_type={summary['losses_by_piece_type']}")
    print(f"piece_loss_frequency_per_game={summary['piece_loss_frequency_per_game']}")
    print(f"rescue_rate={summary['rescue_rate']:.3f} rescues={summary['rescues']}/{summary['rescue_opportunities']}")
    print(f"avg_pressure={summary['avg_pressure']}")
    print(f"tuning={summary['tuning']}")
    print(f"avg_final_material_balance_red={summary['avg_final_material_balance_red']:.2f}")
    print(f"underdog_wins={summary['underdog_wins']}")
    print(f"wins_by_final_piece_count={summary['wins_by_final_piece_count']}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    engine = RustEngine()
    policy: PolicyMode = args.policy
    input_view = args.input_view or "base"
    if policy == "model":
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

    if args.session:
        if policy != "model":
            raise ValueError("--session eval currently supports --policy model")
        if getattr(model, "architecture", None) in ("sequence_transformer", "encoder_sequence"):
            raise ValueError(f"{model.architecture} eval requires Python full-policy history; omit --session and add --full-policy.")
        assert model is not None
        records = session_model_selfplay_records(
            engine,
            model,
            games=args.games,
            max_plies=args.max_plies,
            seed=args.seed,
            temperature=args.temperature,
            device=device,
            batch_size=args.rollout_batch_size,
            record_samples=False,
            collect_metrics=not args.no_pressure,
            input_view=input_view,
        )
    else:
        records = selfplay_records(
            engine,
            games=args.games,
            max_plies=args.max_plies,
            seed=args.seed,
            policy=policy,
            model=model,
            temperature=args.temperature,
            device=device,
            input_view=input_view,
            full_policy=args.full_policy,
            max_tuning_actions=args.max_tuning_actions,
            history_plies=getattr(model, "history_plies", 1),
        )
    summary = aggregate(records)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_summary(summary)


if __name__ == "__main__":
    main()
