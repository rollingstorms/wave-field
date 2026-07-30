from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch

from .engine import RustEngine
from .model import PolicyValueNet
from .selfplay import PolicyMode, selfplay_records, session_model_selfplay_records
from .train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Wave Field self-play policies.")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--policy", choices=("random", "model", "heuristic"), default="random")
    parser.add_argument("--checkpoint", type=Path, default=Path("training/checkpoints/policy_value.pt"))
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--session", action="store_true", help="Use fast Rust session rollout for model eval.")
    parser.add_argument("--rollout-batch-size", type=int, default=128)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def load_model(path: Path, hidden_size: int, device: torch.device) -> PolicyValueNet:
    model = PolicyValueNet(hidden_size=hidden_size).to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def aggregate(records: List[Any]) -> Dict[str, Any]:
    winners = Counter(record.stats.winner or "capped" for record in records)
    first_loss_winner = Counter()
    first_loss_loser = Counter()
    losses_by_piece_type = Counter()
    losses_by_player = Counter()
    wins_by_piece_count: Dict[str, Counter[int]] = defaultdict(Counter)
    underdog_wins = Counter()
    pressure_totals = Counter()
    rescue_opportunities = 0
    rescues = 0

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

        if winner:
            winner_count = stats.final_piece_counts[winner]
            loser = "blue" if winner == "red" else "red"
            loser_count = stats.final_piece_counts[loser]
            wins_by_piece_count[winner][winner_count] += 1
            if winner_count < loser_count:
                underdog_wins[winner] += 1

    pressure_samples = sum(record.stats.pressure_samples for record in records)
    return {
        "games": len(records),
        "decisive": sum(1 for record in records if record.stats.decisive),
        "capped": sum(1 for record in records if not record.stats.decisive),
        "mean_plies": sum(record.stats.plies for record in records) / len(records) if records else 0.0,
        "wins": dict(winners),
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
            for player in ("red", "blue")
        },
        "underdog_wins": dict(underdog_wins),
        "wins_by_final_piece_count": {
            player: dict(counts) for player, counts in wins_by_piece_count.items()
        },
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print(f"games={summary['games']} decisive={summary['decisive']} capped={summary['capped']}")
    print(f"mean_plies={summary['mean_plies']:.2f} wins={summary['wins']}")
    print(f"first_loss_team_won={summary['first_loss_team_won']}")
    print(f"first_loss_team_lost={summary['first_loss_team_lost']}")
    print(f"losses_by_player={summary['losses_by_player']}")
    print(f"losses_by_piece_type={summary['losses_by_piece_type']}")
    print(f"piece_loss_frequency_per_game={summary['piece_loss_frequency_per_game']}")
    print(f"rescue_rate={summary['rescue_rate']:.3f} rescues={summary['rescues']}/{summary['rescue_opportunities']}")
    print(f"avg_pressure={summary['avg_pressure']}")
    print(f"underdog_wins={summary['underdog_wins']}")
    print(f"wins_by_final_piece_count={summary['wins_by_final_piece_count']}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    engine = RustEngine()
    policy: PolicyMode = args.policy
    model = load_model(args.checkpoint, args.hidden_size, device) if policy == "model" else None

    if args.session:
        if policy != "model":
            raise ValueError("--session eval currently supports --policy model")
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
            collect_metrics=True,
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
        )
    summary = aggregate(records)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_summary(summary)


if __name__ == "__main__":
    main()
