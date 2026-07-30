from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .eval import aggregate
from .engine import RustEngine
from .model import PolicyValueNet
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
    parser.add_argument("--model-games", type=int, default=100)
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
    legal_counts = [
        int(sample.metadata["legal_count"])
        for sample in samples
        if "legal_count" in sample.metadata
    ]
    summary: Dict[str, Any] = {
        "sources": dict(sources),
        "phases": dict(phases),
    }
    if legal_counts:
        summary["legal_count"] = {
            "mean": float(np.mean(legal_counts)),
            "min": int(np.min(legal_counts)),
            "max": int(np.max(legal_counts)),
        }
    return summary


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
) -> None:
    started_at = time.perf_counter()
    records = session_model_selfplay_records(
        engine,
        model,
        games=args.eval_games,
        max_plies=args.max_plies,
        seed=args.seed + 10_000 + iteration,
        temperature=args.eval_temperature,
        device=device,
        batch_size=args.rollout_batch_size,
        record_samples=False,
        collect_metrics=args.eval_pressure,
    )
    logger.write(
        {
            "event": "eval",
            "iteration": iteration,
            "seconds": round(time.perf_counter() - started_at, 3),
            "summary": aggregate(records),
        }
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
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
        train_samples(
            model,
            optimizer,
            samples,
            device,
            args.batch_size,
            args.epochs,
            "rust_random",
            0,
            logger,
        )

    for iteration in range(1, args.iterations + 1):
        profile: Dict[str, float] = {}
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
        samples = [sample for record in records for sample in record.samples]
        logger.write(
            {
                "event": "generate",
                "phase": "model_session",
                "iteration": iteration,
                "seconds": round(time.perf_counter() - started_at, 3),
                "summary": generation_summary(records, samples),
                "profile": {key: round(value, 6) for key, value in sorted(profile.items())},
            }
        )
        train_samples(
            model,
            optimizer,
            samples,
            device,
            args.batch_size,
            args.epochs,
            "model_session",
            iteration,
            logger,
        )
        if args.eval_every > 0 and iteration % args.eval_every == 0:
            run_session_eval(engine, model, args, device, iteration, logger)

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
