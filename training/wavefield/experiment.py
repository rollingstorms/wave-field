from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch

from .eval import aggregate
from .encoding import SIDE_SIZE, board_channels_for_view
from .engine import RustEngine
from .model import PolicyValueNet
from .scenarios import DEFAULT_SCENARIOS, build_scenario_states, scenario_names
from .selfplay import Sample, heuristic_bootstrap_records, rust_random_training_samples, session_model_selfplay_records
from .train import resolve_device, samples_to_tensors, train_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run logged Wave Field training experiments.")
    parser.add_argument("--run-dir", type=Path, default=Path("training/runs/dev"))
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--model-arch", choices=("conv", "residual", "transformer"), default="conv")
    parser.add_argument("--input-view", choices=("base", "piece_identity"), default="base")
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--pretrain-random-games", type=int, default=0)
    parser.add_argument("--heuristic-bootstrap-games", type=int, default=0)
    parser.add_argument("--heuristic-bootstrap-per-iteration", type=int, default=0)
    parser.add_argument("--random-games-per-iteration", type=int, default=0)
    parser.add_argument("--model-games", type=int, default=100)
    parser.add_argument("--scenario-games-per-iteration", type=int, default=0)
    parser.add_argument("--scenario-bootstrap-per-iteration", type=int, default=0)
    parser.add_argument("--scenario-eval-games", type=int, default=0)
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--rollout-batch-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--full-policy", action="store_true", help="Train model turns with learned tune/move heads.")
    parser.add_argument("--max-tuning-actions", type=int, default=3)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-games", type=int, default=25)
    parser.add_argument("--eval-max-plies", type=int, default=None, help="Override --max-plies for periodic eval.")
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-pressure", action="store_true")
    parser.add_argument("--baseline-eval-games", type=int, default=0, help="Run side-swapped model-vs-baseline matches during eval.")
    parser.add_argument("--baseline-eval-max-plies", type=int, default=None, help="Override ply cap for baseline eval matches.")
    parser.add_argument("--baseline-opponents", default="heuristic", help="Comma-separated policies: heuristic,random.")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--cap-value", choices=("zero", "material"), default="material")
    parser.add_argument("--progress", action="store_true", help="Show compact ANSI progress lines during long runs.")
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


class TerminalProgress:
    def __init__(self, enabled: bool, iterations: int) -> None:
        self.enabled = enabled
        self.iterations = iterations
        self.active = False

    def clear(self) -> None:
        if not self.enabled or not self.active:
            return
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()
        self.active = False

    def line(self, text: str) -> None:
        if not self.enabled:
            return
        sys.stderr.write(f"\r\033[K{text}")
        sys.stderr.flush()
        self.active = True

    def finish(self, text: str) -> None:
        if not self.enabled:
            return
        self.line(text)
        sys.stderr.write("\n")
        sys.stderr.flush()
        self.active = False

    def phase(self, iteration: int, phase: str, detail: str = "") -> None:
        prefix = f"iter {iteration}/{self.iterations}" if iteration > 0 else "bootstrap"
        suffix = f" {detail}" if detail else ""
        self.line(f"{prefix} {phase}{suffix}")

    def epoch(
        self,
        iteration: int,
        phase: str,
        epoch: int,
        epochs: int,
        batch: int,
        batches: int,
        totals: Dict[str, float],
    ) -> None:
        prefix = f"iter {iteration}/{self.iterations}" if iteration > 0 else "bootstrap"
        self.line(f"{prefix} {phase} epoch {epoch}/{epochs} batch {batch}/{batches} loss {totals['loss']:.4f}")


class JsonlLogger:
    def __init__(self, path: Path, progress: TerminalProgress | None = None) -> None:
        self.path = path
        self.progress = progress
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def write(self, event: Dict[str, Any]) -> None:
        if self.progress is not None:
            self.progress.clear()
        print(json.dumps(event, sort_keys=True), flush=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


def serializable_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


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
    legal_tuning_counts = [
        int(sample.metadata["legal_tuning_count"])
        for sample in samples
        if "legal_tuning_count" in sample.metadata
    ]
    material_balances = [
        int(sample.metadata["material_balance_current"])
        for sample in samples
        if "material_balance_current" in sample.metadata
    ]
    values = [float(sample.value) for sample in samples]
    action_kinds = Counter("tune" if sample.action_kind == 1 else "move" for sample in samples)
    summary: Dict[str, Any] = {
        "sources": dict(sources),
        "phases": dict(phases),
        "scenarios": dict(scenarios),
        "low_material": low_material,
        "action_kinds": dict(action_kinds),
    }
    if legal_counts:
        summary["legal_count"] = numeric_summary(legal_counts)
    if legal_tuning_counts:
        summary["legal_tuning_count"] = numeric_summary(legal_tuning_counts)
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
    progress: TerminalProgress,
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
        losses = train_epoch(
            model,
            optimizer,
            tensors,
            batch_size=batch_size,
            progress=lambda batch, batches, totals: progress.epoch(
                iteration,
                phase,
                epoch,
                epochs,
                batch,
                batches,
                totals,
            ),
        )
        progress.clear()
        logger.write(
            {
                "event": "train_epoch",
                "phase": phase,
                "iteration": iteration,
                "epoch": epoch,
                "samples": len(samples),
                "seconds": round(time.perf_counter() - started_at, 3),
                "loss": round(losses["loss"], 6),
                "kind": round(losses["kind"], 6),
                "policy": round(losses["policy"], 6),
                "tuning": round(losses["tuning"], 6),
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
    progress: TerminalProgress,
    phase: str = "eval",
    initial_states: List[Dict[str, Any]] | None = None,
) -> None:
    started_at = time.perf_counter()
    games = len(initial_states) if initial_states is not None else args.eval_games
    max_plies = args.eval_max_plies or args.max_plies
    progress.phase(iteration, phase, f"{games} games")
    if args.full_policy:
        from .selfplay import selfplay_records
        records = selfplay_records(
            engine,
            games=games,
            max_plies=max_plies,
            seed=args.seed + 10_000 + iteration,
            policy="model",
            model=model,
            temperature=args.eval_temperature,
            device=device,
            cap_value=args.cap_value,
            collect_metrics=args.eval_pressure,
            record_samples=False,
            input_view=args.input_view,
            full_policy=True,
            max_tuning_actions=args.max_tuning_actions,
            initial_states=initial_states,
        )
    else:
        records = session_model_selfplay_records(
            engine,
            model,
            games=games,
            max_plies=max_plies,
            seed=args.seed + 10_000 + iteration,
            temperature=args.eval_temperature,
            device=device,
            batch_size=args.rollout_batch_size,
            record_samples=False,
            collect_metrics=args.eval_pressure,
            initial_states=initial_states,
            input_view=args.input_view,
        )
    logger.write(
        {
            "event": phase,
            "iteration": iteration,
            "seconds": round(time.perf_counter() - started_at, 3),
            "max_plies": max_plies,
            "summary": aggregate(records),
        }
    )


def parse_baseline_opponents(raw: str) -> List[str]:
    opponents = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [opponent for opponent in opponents if opponent not in {"heuristic", "random"}]
    if invalid:
        raise ValueError(f"Invalid baseline opponent(s): {invalid}. Expected heuristic or random.")
    return opponents


def run_baseline_eval(
    engine: RustEngine,
    model: PolicyValueNet,
    args: argparse.Namespace,
    device: torch.device,
    iteration: int,
    logger: JsonlLogger,
    progress: TerminalProgress,
    opponents: List[str],
) -> None:
    if args.baseline_eval_games <= 0 or not opponents:
        return

    from .match import play_match_game

    max_plies = args.baseline_eval_max_plies or args.eval_max_plies or args.max_plies
    games_per_side = args.baseline_eval_games
    for opponent in opponents:
        for model_side in ("blue", "red"):
            started_at = time.perf_counter()
            policies = {
                "blue": "model" if model_side == "blue" else opponent,
                "red": "model" if model_side == "red" else opponent,
            }
            progress.phase(
                iteration,
                "baseline_eval",
                f"model={model_side} vs {opponent} {games_per_side} games",
            )
            records = [
                play_match_game(
                    engine,
                    policies=policies,
                    model=model,
                    device=device,
                    max_plies=max_plies,
                    seed=args.seed + 90_000 + iteration * 1_000 + game,
                    temperature=args.eval_temperature,
                    input_view=args.input_view,
                    full_policy=args.full_policy,
                    max_tuning_actions=args.max_tuning_actions,
                    collect_metrics=args.eval_pressure,
                )
                for game in range(games_per_side)
            ]
            logger.write(
                {
                    "event": "baseline_eval",
                    "iteration": iteration,
                    "opponent": opponent,
                    "model_side": model_side,
                    "policies": policies,
                    "games_per_side": games_per_side,
                    "max_plies": max_plies,
                    "seconds": round(time.perf_counter() - started_at, 3),
                    "summary": aggregate(records),
                }
            )


def save_checkpoint(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    logger: JsonlLogger,
    iteration: int,
) -> None:
    checkpoint_path = args.run_dir / ("checkpoint.pt" if iteration == args.iterations else f"checkpoint-iter-{iteration}.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "hidden_size": args.hidden_size,
            "board_channels": board_channels_for_view(args.input_view),
            "side_size": SIDE_SIZE,
            "model_arch": args.model_arch,
            "input_view": args.input_view,
            "iteration": iteration,
            "config": serializable_config(args),
        },
        checkpoint_path,
    )
    logger.write({"event": "saved", "iteration": iteration, "checkpoint": str(checkpoint_path)})


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    source_weights = parse_weights(args.source_weights)
    phase_weights = parse_weights(args.phase_weights)
    scenarios = scenario_names(args.scenarios)
    baseline_opponents = parse_baseline_opponents(args.baseline_opponents)
    if args.input_view != "base" and (args.pretrain_random_games > 0 or args.random_games_per_iteration > 0):
        raise ValueError("Rust random training batches currently support only --input-view base")
    torch.manual_seed(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    progress = TerminalProgress(args.progress, args.iterations)
    logger = JsonlLogger(args.run_dir / "events.jsonl", progress)

    engine = RustEngine()
    model = PolicyValueNet(
        hidden_size=args.hidden_size,
        board_channels=board_channels_for_view(args.input_view),
        side_size=SIDE_SIZE,
        architecture=args.model_arch,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    logger.write(
        {
            "event": "config",
            "config": {
                **serializable_config(args),
                "run_dir": str(args.run_dir),
                "device": str(device),
                "source_weights": source_weights,
                "phase_weights": phase_weights,
                "scenarios": scenarios,
                "board_channels": board_channels_for_view(args.input_view),
                "side_size": SIDE_SIZE,
            },
        }
    )

    if args.pretrain_random_games > 0:
        started_at = time.perf_counter()
        progress.phase(0, "rust_random", f"{args.pretrain_random_games} games")
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
            progress,
        )

    if args.heuristic_bootstrap_games > 0:
        started_at = time.perf_counter()
        progress.phase(0, "heuristic_bootstrap", f"{args.heuristic_bootstrap_games} games")
        records = heuristic_bootstrap_records(
            engine,
            games=args.heuristic_bootstrap_games,
            max_plies=args.max_plies,
            seed=args.seed + 25_000,
            cap_value=args.cap_value,
            input_view=args.input_view,
            collect_metrics=False,
        )
        samples = [sample for record in records for sample in record.samples]
        logger.write(
            {
                "event": "generate",
                "phase": "heuristic_bootstrap",
                "iteration": 0,
                "seconds": round(time.perf_counter() - started_at, 3),
                "summary": generation_summary(records, samples),
            }
        )
        weighted_samples = replay_weight_samples(samples, source_weights, phase_weights)
        if weighted_samples is not samples:
            logger.write(
                {
                    "event": "augment",
                    "phase": "heuristic_bootstrap",
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
            "heuristic_bootstrap",
            0,
            logger,
            progress,
        )

    for iteration in range(1, args.iterations + 1):
        iteration_samples: List[Sample] = []
        if args.heuristic_bootstrap_per_iteration > 0:
            started_at = time.perf_counter()
            progress.phase(iteration, "heuristic_bootstrap_iteration", f"{args.heuristic_bootstrap_per_iteration} games")
            records = heuristic_bootstrap_records(
                engine,
                games=args.heuristic_bootstrap_per_iteration,
                max_plies=args.max_plies,
                seed=args.seed + 35_000 + iteration,
                cap_value=args.cap_value,
                input_view=args.input_view,
                collect_metrics=False,
            )
            bootstrap_samples = [sample for record in records for sample in record.samples]
            iteration_samples.extend(bootstrap_samples)
            logger.write(
                {
                    "event": "generate",
                    "phase": "heuristic_bootstrap_iteration",
                    "iteration": iteration,
                    "seconds": round(time.perf_counter() - started_at, 3),
                    "summary": generation_summary(records, bootstrap_samples),
                }
            )

        if args.random_games_per_iteration > 0:
            started_at = time.perf_counter()
            progress.phase(iteration, "rust_random_iteration", f"{args.random_games_per_iteration} games")
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
            progress.phase(iteration, "model_session", f"{args.model_games} games")
            if args.full_policy:
                from .selfplay import selfplay_records
                records = selfplay_records(
                    engine,
                    games=args.model_games,
                    max_plies=args.max_plies,
                    seed=args.seed + iteration,
                    policy="model",
                    model=model,
                    temperature=args.temperature,
                    device=device,
                    cap_value=args.cap_value,
                    collect_metrics=False,
                    input_view=args.input_view,
                    full_policy=True,
                    max_tuning_actions=args.max_tuning_actions,
                )
            else:
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
                    input_view=args.input_view,
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

        if args.scenario_bootstrap_per_iteration > 0:
            progress.phase(iteration, "scenario_build", f"{args.scenario_bootstrap_per_iteration} games")
            scenario_states = build_scenario_states(
                engine,
                scenarios,
                games=args.scenario_bootstrap_per_iteration,
                seed=args.seed + 65_000 + iteration,
            )
            started_at = time.perf_counter()
            progress.phase(iteration, "scenario_heuristic_bootstrap", f"{args.scenario_bootstrap_per_iteration} games")
            records = heuristic_bootstrap_records(
                engine,
                games=args.scenario_bootstrap_per_iteration,
                max_plies=args.max_plies,
                seed=args.seed + 66_000 + iteration,
                cap_value=args.cap_value,
                input_view=args.input_view,
                initial_states=scenario_states,
                collect_metrics=False,
            )
            bootstrap_samples = [sample for record in records for sample in record.samples]
            iteration_samples.extend(bootstrap_samples)
            logger.write(
                {
                    "event": "generate",
                    "phase": "scenario_heuristic_bootstrap",
                    "iteration": iteration,
                    "seconds": round(time.perf_counter() - started_at, 3),
                    "summary": generation_summary(records, bootstrap_samples),
                }
            )

        if args.scenario_games_per_iteration > 0:
            progress.phase(iteration, "scenario_build", f"{args.scenario_games_per_iteration} games")
            scenario_states = build_scenario_states(
                engine,
                scenarios,
                games=args.scenario_games_per_iteration,
                seed=args.seed + 75_000 + iteration,
            )
            scenario_profile: Dict[str, float] = {}
            started_at = time.perf_counter()
            progress.phase(iteration, "scenario_model_session", f"{args.scenario_games_per_iteration} games")
            if args.full_policy:
                from .selfplay import selfplay_records
                scenario_records = selfplay_records(
                    engine,
                    games=args.scenario_games_per_iteration,
                    max_plies=args.max_plies,
                    seed=args.seed + 80_000 + iteration,
                    policy="model",
                    model=model,
                    temperature=args.temperature,
                    device=device,
                    cap_value=args.cap_value,
                    collect_metrics=False,
                    input_view=args.input_view,
                    full_policy=True,
                    max_tuning_actions=args.max_tuning_actions,
                    initial_states=scenario_states,
                )
            else:
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
                    input_view=args.input_view,
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
            progress,
        )
        if args.eval_every > 0 and iteration % args.eval_every == 0:
            run_session_eval(engine, model, args, device, iteration, logger, progress)
            run_baseline_eval(engine, model, args, device, iteration, logger, progress, baseline_opponents)
            if args.scenario_eval_games > 0:
                progress.phase(iteration, "scenario_eval_build", f"{args.scenario_eval_games} games")
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
                    progress,
                    phase="scenario_eval",
                    initial_states=scenario_eval_states,
                )
        if args.save_every > 0 and iteration % args.save_every == 0:
            save_checkpoint(model, optimizer, args, logger, iteration)

    if args.iterations == 0 or args.save_every <= 0 or args.iterations % args.save_every != 0:
        save_checkpoint(model, optimizer, args, logger, args.iterations)
    engine.close()


if __name__ == "__main__":
    main()
