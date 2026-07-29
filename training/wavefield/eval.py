from __future__ import annotations

import argparse

from .engine import RustEngine
from .selfplay import random_selfplay_game


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small random self-play evaluation smoke test.")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--seed", type=int, default=90210)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = RustEngine()
    samples = 0
    decisive = 0
    for game in range(args.games):
        game_samples = random_selfplay_game(engine, max_plies=args.max_plies, seed=args.seed + game)
        samples += len(game_samples)
        if game_samples and game_samples[-1].value != 0.0:
            decisive += 1
    print({"games": args.games, "samples": samples, "decisive": decisive})


if __name__ == "__main__":
    main()
