from __future__ import annotations

import argparse
import time
from typing import Any, Dict

import torch

from .engine import RustEngine, load_initial_state
from .model import PolicyValueNet
from .selfplay import batched_model_selfplay_records, selfplay_records, session_model_selfplay_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Wave Field self-play paths.")
    parser.add_argument("--games", type=int, default=25)
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--build-engine", action="store_true")
    parser.add_argument("--include-model", action="store_true")
    parser.add_argument("--profile-model", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--rollout-batch-size", type=int, default=32)
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

    items = [rust, rust_training, rust_training_profile, python]

    if args.include_model:
        torch.manual_seed(args.seed)
        model = PolicyValueNet(hidden_size=args.hidden_size)
        legacy_profile: Dict[str, float] = {}
        session_profile: Dict[str, float] = {}
        legacy_model = timed(
            "python_batched_model_full_state",
            lambda: batched_model_selfplay_records(
                engine,
                model,
                args.games,
                args.max_plies,
                args.seed,
                temperature=0.0,
                batch_size=args.rollout_batch_size,
                profile=legacy_profile if args.profile_model else None,
            ),
        )
        session_model = timed(
            "rust_session_model",
            lambda: session_model_selfplay_records(
                engine,
                model,
                args.games,
                args.max_plies,
                args.seed,
                temperature=0.0,
                batch_size=args.rollout_batch_size,
                profile=session_profile if args.profile_model else None,
            ),
        )
        if args.profile_model:
            legacy_model["profile"] = legacy_profile
            session_model["profile"] = session_profile
        items.extend([legacy_model, session_model])

    for item in items:
        games_per_second = args.games / item["seconds"] if item["seconds"] else 0.0
        print(f"{item['label']} seconds={item['seconds']:.3f} games_per_second={games_per_second:.2f}")
        if item["label"] == "rust_random_training_profile":
            print(item["result"])
        if "profile" in item:
            print(item["profile"])


if __name__ == "__main__":
    main()
