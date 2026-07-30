from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch

from .eval import aggregate
from .engine import RustEngine
from .model import PolicyValueNet
from .scenarios import DEFAULT_SCENARIOS, build_scenario_states, scenario_names
from .selfplay import Sample, rust_random_training_samples, session_model_selfplay_records
from .train import resolve_device, samples_to_tensors, train_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run logged Wave Field training experiments.")
    parser.add_argument("--run-dir", type=Path, default=Path("training/runs/dev"))
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--pretrain-random-games", type=int, default=0)
    parser.add_argument("--random-games-per-iteration", type=int, default=0)
    parser.add_argument("--model-games", type=int, default=100)
    parser.add_argument("--scenario-games-per-iteration", type=int, default=0)
    parser.add_argument("--scenario-eval-games", type=int, default=0)
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--rollout-batch-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-games", type=int, default=25)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-pressure", action="store_true")
    parser.add_argument("--cap-value", choices=("zero", "material"), default="material")
    parser.add_argument(
        "--phase-weights",
        default="",
        help="Comma-separated replay weights, for example opening=1,midgame=1,endgame=2.",
    )
    parser.add_argument(
        "--source-weights",
        default="",
        help="Comma-separated replay weights, for example rust_random=1,rust_session_model=2.",
    )
    return parser.parse_args()


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def write(self, event: Dict[str, Any]) -> None:
        print(json.dumps(event, sort_keys=True), flush=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


def sample_metadata_summary(samples: List[Sample]) -> Dict[str, Any]:
    sources = Counter(str(sample.metadata.get("source", "unknown")) for sample in samples)
    phases = Counter(str(sample.metadata.get("phase", "unknown")) for sample in samples)
    scenarios = Counter(str(sample.metadata.get("scenario", "unknown")) for sample in samples)
    low_material = sum(1 for sample in samples if bool(sample.metadata.get("low_material", False)))
    legal_counts = [
        int(sample.metadata["legal_count"])
        for sample in samples
        if "legal_count" in sample.metadata
    ]
    material_balances = [
        int(sample.metadata["material_balance_current"])
        for sample in samples
        if "material_balance_current" in sample.metadata
    ]
    values = [float(sample.value) for sample in samples]
    summary: Dict[str, Any] = {
        "sources": dict(sources),
        "phases": dict(phases),
        "scenarios": dict(scenarios),
        "low_material": low_material,
    }
    if legal_counts:
        summary["legal_count"] = numeric_summary(legal_counts)
    if material_balances:
        summary["material_balance_current"] = numeric_summary(material_balances)
    if values:
        summary["value"] = numeric_summary(values)
    return summary


def numeric_summary(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float32)
    return {
        "mean": float(np.mean(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def parse_weights(raw: str) -> Dict[str, int]:
    if not raw:
        return {}
    weights: Dict[str, int] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid weight '{item}'. Expected name=integer.")
        name, value = item.split("=", 1)
        weight = int(value)
        if weight < 0:
            raise ValueError(f"Invalid weight '{item}'. Weight must be non-negative.")
        weights[name.strip()] = weight
    return weights


def replay_weight_samples(
    samples: List[Sample],
    source_weights: Dict[str, int],
    phase_weights: Dict[str, int],
) -> List[Sample]:
    if not source_weights and not phase_weights:
        return samples

    weighted: List[Sample] = []
    for sample in samples:
        source = str(sample.metadata.get("source", "unknown"))
        phase = str(sample.metadata.get("phase", "unknown"))
        source_weight = source_weights.get(source, 1)
        phase_weight = phase_weights.get(phase, 1)
        weighted.extend([sample] * (source_weight * phase_weight))
    return weighted


def generation_summary(records: List[Any], samples: List[Sample]) -> Dict[str, Any]:
    return {
        "games": len(records),
        "samples": len(samples),
        "decisive": sum(1 for record in records if record.stats.decisive),
        "mean_plies": (
            sum(record.stats.plies for record in records) / len(records)
            if records
            else 0.0
        ),
        "wins": {
            winner: sum(1 for record in records if (record.stats.winner or "capped") == winner)
            for winner in ("red", "blue", "capped")
        },
        "metadata": sample_metadata_summary(samples),
    }


def train_samples(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    samples: List[Sample],
    device: torch.device,
    batch_size: int,
    epochs: int,
    phase: str,
    iteration: int,
    logger: JsonlLogger,
) -> None:
    if not samples:
        logger.write(
            {
                "event": "train_skipped",
                "phase": phase,
                "iteration": iteration,
                "reason": "no_samples",
            }
        )
        return

    tensors = samples_to_tensors(samples, device)
    for epoch in range(1, epochs + 1):
        started_at = time.perf_counter()
        losses = train_epoch(model, optimizer, tensors, batch_size=batch_size)
        logger.write(
            {
                "event": "train_epoch",
                "phase": phase,
                "iteration": iteration,
                "epoch": epoch,
                "samples": len(samples),
                "seconds": round(time.perf_counter() - started_at, 3),
                "loss": round(losses["loss"], 6),
                "policy": round(losses["policy"], 6),
                "value": round(losses["value"], 6),
            }
        )


def run_session_eval(
    engine: RustEngine,
    model: PolicyValueNet,
    args: argparse.Namespace,
    device: torch.device,
    iteration: int,
    logger: JsonlLogger,
    phase: str = "eval",
    initial_states: List[Dict[str, Any]] | None = None,
) -> None:
    started_at = time.perf_counter()
    games = len(initial_states) if initial_states is not None else args.eval_games
    records = session_model_selfplay_records(
        engine,
        model,
        games=games,
        max_plies=args.max_plies,
        seed=args.seed + 10_000 + iteration,
        temperature=args.eval_temperature,
        device=device,
        batch_size=args.rollout_batch_size,
        record_samples=False,
        collect_metrics=args.eval_pressure,
        initial_states=initial_states,
    )
    logger.write(
        {
            "event": phase,
            "iteration": iteration,
            "seconds": round(time.perf_counter() - started_at, 3),
            "summary": aggregate(records),
        }
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    source_weights = parse_weights(args.source_weights)
    phase_weights = parse_weights(args.phase_weights)
    scenarios = scenario_names(args.scenarios)
    torch.manual_seed(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(args.run_dir / "events.jsonl")

    engine = RustEngine()
    model = PolicyValueNet(hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    logger.write(
        {
            "event": "config",
            "config": {
                **vars(args),
                "run_dir": str(args.run_dir),
                "device": str(device),
                "source_weights": source_weights,
                "phase_weights": phase_weights,
                "scenarios": scenarios,
            },
        }
    )

    if args.pretrain_random_games > 0:
        started_at = time.perf_counter()
        samples, batch_summary = rust_random_training_samples(
            engine,
            games=args.pretrain_random_games,
            max_plies=args.max_plies,
            seed=args.seed,
            cap_value=args.cap_value,
        )
        logger.write(
            {
                "event": "generate",
                "phase": "rust_random",
                "iteration": 0,
                "seconds": round(time.perf_counter() - started_at, 3),
                "summary": {
                    **batch_summary,
                    "metadata": sample_metadata_summary(samples),
                },
            }
        )
        weighted_samples = replay_weight_samples(samples, source_weights, phase_weights)
        if weighted_samples is not samples:
            logger.write(
                {
                    "event": "augment",
                    "phase": "rust_random",
                    "iteration": 0,
                    "raw_samples": len(samples),
                    "training_samples": len(weighted_samples),
                    "summary": sample_metadata_summary(weighted_samples),
                }
            )
        train_samples(
            model,
            optimizer,
            weighted_samples,
            device,
            args.batch_size,
            args.epochs,
            "rust_random",
            0,
            logger,
        )

    for iteration in range(1, args.iterations + 1):
        iteration_samples: List[Sample] = []
        if args.random_games_per_iteration > 0:
            started_at = time.perf_counter()
            random_samples, random_summary = rust_random_training_samples(
                engine,
                games=args.random_games_per_iteration,
                max_plies=args.max_plies,
                seed=args.seed + 50_000 + iteration,
                cap_value=args.cap_value,
            )
            iteration_samples.extend(random_samples)
            logger.write(
                {
                    "event": "generate",
                    "phase": "rust_random_iteration",
                    "iteration": iteration,
                    "seconds": round(time.perf_counter() - started_at, 3),
                    "summary": {
                        **random_summary,
                        "metadata": sample_metadata_summary(random_samples),
                    },
                }
            )

        profile: Dict[str, float] = {}
        if args.model_games > 0:
            started_at = time.perf_counter()
            records = session_model_selfplay_records(
                engine,
                model,
                games=args.model_games,
                max_plies=args.max_plies,
                seed=args.seed + iteration,
                temperature=args.temperature,
                device=device,
                cap_value=args.cap_value,
                batch_size=args.rollout_batch_size,
                profile=profile,
            )
            model_samples = [sample for record in records for sample in record.samples]
            iteration_samples.extend(model_samples)
            logger.write(
                {
                    "event": "generate",
                    "phase": "model_session",
                    "iteration": iteration,
                    "seconds": round(time.perf_counter() - started_at, 3),
                    "summary": generation_summary(records, model_samples),
                    "profile": {key: round(value, 6) for key, value in sorted(profile.items())},
                }
            )

        if args.scenario_games_per_iteration > 0:
            scenario_states = build_scenario_states(
                engine,
                scenarios,
                games=args.scenario_games_per_iteration,
                seed=args.seed + 75_000 + iteration,
            )
            scenario_profile: Dict[str, float] = {}
            started_at = time.perf_counter()
            scenario_records = session_model_selfplay_records(
                engine,
                model,
                games=args.scenario_games_per_iteration,
                max_plies=args.max_plies,
                seed=args.seed + 80_000 + iteration,
                temperature=args.temperature,
                device=device,
                cap_value=args.cap_value,
                batch_size=args.rollout_batch_size,
                profile=scenario_profile,
                initial_states=scenario_states,
            )
            scenario_samples = [sample for record in scenario_records for sample in record.samples]
            iteration_samples.extend(scenario_samples)
            logger.write(
                {
                    "event": "generate",
                    "phase": "scenario_model_session",
                    "iteration": iteration,
                    "seconds": round(time.perf_counter() - started_at, 3),
                    "summary": generation_summary(scenario_records, scenario_samples),
                    "profile": {key: round(value, 6) for key, value in sorted(scenario_profile.items())},
                }
            )

        training_samples = replay_weight_samples(iteration_samples, source_weights, phase_weights)
        logger.write(
            {
                "event": "train_batch",
                "phase": "curriculum",
                "iteration": iteration,
                "raw_samples": len(iteration_samples),
                "training_samples": len(training_samples),
                "summary": sample_metadata_summary(training_samples),
            }
        )
        train_samples(
            model,
            optimizer,
            training_samples,
            device,
            args.batch_size,
            args.epochs,
            "curriculum",
            iteration,
            logger,
        )
        if args.eval_every > 0 and iteration % args.eval_every == 0:
            run_session_eval(engine, model, args, device, iteration, logger)
            if args.scenario_eval_games > 0:
                scenario_eval_states = build_scenario_states(
                    engine,
                    scenarios,
                    games=args.scenario_eval_games,
                    seed=args.seed + 90_000 + iteration,
                )
                run_session_eval(
                    engine,
                    model,
                    args,
                    device,
                    iteration,
                    logger,
                    phase="scenario_eval",
                    initial_states=scenario_eval_states,
                )

    checkpoint_path = args.run_dir / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "hidden_size": args.hidden_size,
            "config": vars(args),
        },
        checkpoint_path,
    )
    logger.write({"event": "saved", "checkpoint": str(checkpoint_path)})
    engine.close()


if __name__ == "__main__":
    main()
