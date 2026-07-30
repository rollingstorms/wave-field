from __future__ import annotations

import argparse
import time
from typing import Any, Dict

from .engine import RustEngine, load_initial_state
from .selfplay import selfplay_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Wave Field self-play paths.")
    parser.add_argument("--games", type=int, default=25)
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--build-engine", action="store_true")
    return parser.parse_args()


def timed(label: str, fn: Any) -> Dict[str, Any]:
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return {"label": label, "seconds": elapsed, "result": result}


def main() -> None:
    args = parse_args()
    engine = RustEngine()
    if args.build_engine:
        engine.build_release()

    state = load_initial_state()

    rust = timed(
        "rust_random_lean",
        lambda: engine.simulate_random_lean_games(state, args.games, args.max_plies, args.seed),
    )
    rust_training = timed(
        "rust_random_training_batch",
        lambda: engine.generate_random_training_batch(state, args.games, args.max_plies, args.seed),
    )
    rust_training_profile = timed(
        "rust_random_training_profile",
        lambda: engine.profile_random_training_batch(state, args.games, args.max_plies, args.seed),
    )
    python = timed(
        "python_random_with_encoding",
        lambda: selfplay_records(engine, args.games, args.max_plies, args.seed, policy="random"),
    )

    for item in (rust, rust_training, rust_training_profile, python):
        games_per_second = args.games / item["seconds"] if item["seconds"] else 0.0
        print(f"{item['label']} seconds={item['seconds']:.3f} games_per_second={games_per_second:.2f}")
        if item["label"] == "rust_random_training_profile":
            print(item["result"])


if __name__ == "__main__":
    main()
