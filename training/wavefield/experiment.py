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
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument(
        "--model-arch",
        choices=("conv", "residual", "transformer", "sequence_transformer", "encoder_sequence"),
        default="conv",
    )
    parser.add_argument("--input-view", choices=("base", "piece_identity"), default="base")
    parser.add_argument("--history-plies", type=int, default=1, help="State history window for sequence models.")
    parser.add_argument("--encoder-checkpoint", type=Path, default=None, help="State-model checkpoint for encoder_sequence.")
    parser.add_argument("--encoder-arch", choices=("conv", "residual", "transformer"), default=None)
    parser.add_argument("--encoder-hidden-size", type=int, default=None)
    parser.add_argument("--unfreeze-encoder", action="store_true", help="Fine-tune encoder_sequence state encoder.")
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--pretrain-random-games", type=int, default=0)
    parser.add_argument("--bootstrap-policy", choices=("heuristic", "hard", "easy"), default="heuristic")
    parser.add_argument("--bootstrap-variety", type=float, default=None)
    parser.add_argument("--bootstrap-time-budget-ms", type=int, default=None)
    parser.add_argument("--heuristic-bootstrap-games", type=int, default=0)
    parser.add_argument("--heuristic-bootstrap-per-iteration", type=int, default=0)
    parser.add_argument("--random-games-per-iteration", type=int, default=0)
    parser.add_argument("--model-games", type=int, default=100)
    parser.add_argument("--scenario-games-per-iteration", type=int, default=0)
    parser.add_argument("--scenario-bootstrap-per-iteration", type=int, default=0)
    parser.add_argument("--scenario-eval-games", type=int, default=0)
    parser.add_argument("--tactical-eval-games", type=int, default=0)
    parser.add_argument("--tactical-eval-max-plies", type=int, default=1)
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--rollout-batch-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--kind-temperature", type=float, default=None, help="Training self-play temperature for tune-vs-move kind sampling.")
    parser.add_argument("--tuning-temperature", type=float, default=None, help="Training self-play temperature for tuning action sampling.")
    parser.add_argument(
        "--force-first-tune-prob",
        type=float,
        default=0.0,
        help="Training self-play probability of forcing the first eligible full-policy action to be tuning.",
    )
    parser.add_argument("--full-policy", action="store_true", help="Train model turns with learned tune/move heads.")
    parser.add_argument("--max-tuning-actions", type=int, default=3)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-games", type=int, default=25)
    parser.add_argument("--eval-max-plies", type=int, default=None, help="Override --max-plies for periodic eval.")
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-pressure", action="store_true")
    parser.add_argument("--baseline-eval-games", type=int, default=0, help="Run side-swapped model-vs-baseline matches during eval.")
    parser.add_argument("--baseline-eval-max-plies", type=int, default=None, help="Override ply cap for baseline eval matches.")
    parser.add_argument("--baseline-opponents", default="heuristic", help="Comma-separated policies: heuristic,hard,easy,random.")
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


def encoder_config_from_checkpoint(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    if args.model_arch != "encoder_sequence":
        return {}

    checkpoint: Dict[str, Any] = {}
    if args.encoder_checkpoint is not None:
        checkpoint = torch.load(args.encoder_checkpoint, map_location=device, weights_only=False)

    encoder_arch = args.encoder_arch or checkpoint.get("model_arch", "transformer")
    if encoder_arch == "encoder_sequence":
        raise ValueError("encoder_sequence requires a state-only encoder checkpoint or encoder arch.")
    encoder_hidden_size = int(args.encoder_hidden_size or checkpoint.get("hidden_size", args.hidden_size))
    encoder_input_view = checkpoint.get("input_view", args.input_view)
    if encoder_input_view != args.input_view:
        raise ValueError(
            f"encoder checkpoint input_view={encoder_input_view!r} does not match --input-view={args.input_view!r}"
        )
    encoder_board_channels = int(checkpoint.get("board_channels", board_channels_for_view(args.input_view)))
    if encoder_board_channels != board_channels_for_view(args.input_view):
        raise ValueError(
            f"encoder checkpoint board_channels={encoder_board_channels} does not match --input-view={args.input_view!r}"
        )
    return {
        "checkpoint": checkpoint,
        "encoder_arch": encoder_arch,
        "encoder_hidden_size": encoder_hidden_size,
        "freeze_encoder": not args.unfreeze_encoder,
    }


def load_encoder_checkpoint(model: PolicyValueNet, checkpoint: Dict[str, Any]) -> None:
    if not checkpoint:
        return
    if getattr(model, "state_encoder", None) is None:
        raise ValueError("--encoder-checkpoint can only be used with --model-arch encoder_sequence")
    missing, unexpected = model.state_encoder.load_state_dict(checkpoint["model"], strict=False)
    allowed_unexpected = {
        "policy.weight",
        "policy.bias",
        "action_kind.weight",
        "action_kind.bias",
        "tuning_policy.weight",
        "tuning_policy.bias",
        "value.0.weight",
        "value.0.bias",
    }
    unexpected = [key for key in unexpected if key not in allowed_unexpected]
    if unexpected:
        raise ValueError(f"Unexpected encoder checkpoint keys: {unexpected[:5]}")
    if missing:
        raise ValueError(f"Encoder checkpoint is missing keys: {missing[:5]}")


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
    players = Counter(sample.player for sample in samples)
    history_plies = Counter(int(sample.metadata.get("history_plies", 1)) for sample in samples)
    summary: Dict[str, Any] = {
        "sources": dict(sources),
        "phases": dict(phases),
        "scenarios": dict(scenarios),
        "low_material": low_material,
        "action_kinds": dict(action_kinds),
        "players": dict(players),
        "history_plies": dict(history_plies),
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


def bootstrap_variety(args: argparse.Namespace) -> float:
    if args.bootstrap_variety is not None:
        return args.bootstrap_variety
    return 0.0 if args.bootstrap_policy == "hard" else 0.55


def bootstrap_time_budget_ms(args: argparse.Namespace) -> int:
    if args.bootstrap_time_budget_ms is not None:
        return args.bootstrap_time_budget_ms
    return 1_500 if args.bootstrap_policy == "hard" else 10


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
                "history_plies": getattr(model, "history_plies", 1),
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
            history_plies=args.history_plies,
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
    invalid = [opponent for opponent in opponents if opponent not in {"heuristic", "hard", "easy", "random"}]
    if invalid:
        raise ValueError(f"Invalid baseline opponent(s): {invalid}. Expected heuristic, hard, easy, or random.")
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


def run_tactical_eval_phase(
    engine: RustEngine,
    model: PolicyValueNet,
    args: argparse.Namespace,
    device: torch.device,
    iteration: int,
    logger: JsonlLogger,
    progress: TerminalProgress,
    scenarios: List[str],
) -> None:
    if args.tactical_eval_games <= 0:
        return

    from .tactical_eval import run_tactical_eval

    progress.phase(iteration, "tactical_eval", f"{args.tactical_eval_games} games")
    summary = run_tactical_eval(
        engine,
        model,
        device,
        input_view=args.input_view,
        scenarios=scenarios,
        games=args.tactical_eval_games,
        seed=args.seed + 110_000 + iteration,
        max_plies=args.tactical_eval_max_plies,
    )
    logger.write(
        {
            "event": "tactical_eval",
            "iteration": iteration,
            "summary": summary,
        }
    )


def save_checkpoint(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    logger: JsonlLogger,
    iteration: int,
    final_iteration: int,
) -> None:
    checkpoint_path = args.run_dir / ("checkpoint.pt" if iteration == final_iteration else f"checkpoint-iter-{iteration}.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "hidden_size": args.hidden_size,
            "board_channels": board_channels_for_view(args.input_view),
            "side_size": SIDE_SIZE,
            "model_arch": args.model_arch,
            "input_view": args.input_view,
            "history_plies": args.history_plies,
            "encoder_arch": getattr(model, "encoder_arch", None),
            "encoder_hidden_size": getattr(model, "encoder_hidden_size", None),
            "freeze_encoder": getattr(model, "freeze_encoder", None),
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
    teacher_variety = bootstrap_variety(args)
    teacher_time_budget_ms = bootstrap_time_budget_ms(args)
    baseline_opponents = parse_baseline_opponents(args.baseline_opponents)
    if args.input_view != "base" and (args.pretrain_random_games > 0 or args.random_games_per_iteration > 0):
        raise ValueError("Rust random training batches currently support only --input-view base")
    if args.model_arch in ("sequence_transformer", "encoder_sequence") and not args.full_policy:
        raise ValueError(f"{args.model_arch} experiments currently require --full-policy Python rollouts.")
    if args.model_arch in ("sequence_transformer", "encoder_sequence") and args.history_plies < 2:
        raise ValueError(f"{args.model_arch} requires --history-plies >= 2.")
    if args.model_arch in ("sequence_transformer", "encoder_sequence") and (
        args.pretrain_random_games > 0
        or args.random_games_per_iteration > 0
    ):
        raise ValueError(f"{args.model_arch} currently does not support Rust random encoded batches.")
    if args.encoder_checkpoint is not None and args.model_arch != "encoder_sequence":
        raise ValueError("--encoder-checkpoint requires --model-arch encoder_sequence.")
    if args.model_arch == "encoder_sequence" and args.encoder_checkpoint is None and not args.unfreeze_encoder:
        raise ValueError("encoder_sequence needs --encoder-checkpoint unless --unfreeze-encoder trains it from scratch.")
    if not 0.0 <= args.force_first_tune_prob <= 1.0:
        raise ValueError("--force-first-tune-prob must be between 0 and 1.")
    if args.resume_checkpoint is not None and (args.pretrain_random_games > 0 or args.heuristic_bootstrap_games > 0):
        raise ValueError("Resume runs should use per-iteration generation, not bootstrap phases.")
    torch.manual_seed(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    start_iteration = 0
    progress = TerminalProgress(args.progress, args.iterations)
    logger = JsonlLogger(args.run_dir / "events.jsonl", progress)

    engine = RustEngine()
    encoder_config = encoder_config_from_checkpoint(args, device)
    model = PolicyValueNet(
        hidden_size=args.hidden_size,
        board_channels=board_channels_for_view(args.input_view),
        side_size=SIDE_SIZE,
        architecture=args.model_arch,
        history_plies=args.history_plies,
        encoder_arch=encoder_config.get("encoder_arch") or args.encoder_arch or "transformer",
        encoder_hidden_size=encoder_config.get("encoder_hidden_size") or args.encoder_hidden_size,
        freeze_encoder=bool(encoder_config.get("freeze_encoder", not args.unfreeze_encoder)),
    ).to(device)
    load_encoder_checkpoint(model, encoder_config.get("checkpoint", {}))
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr)
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=False)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_iteration = int(checkpoint.get("iteration", 0))
        progress.iterations = start_iteration + args.iterations

    logger.write(
        {
            "event": "config",
            "config": {
                **serializable_config(args),
                "run_dir": str(args.run_dir),
                "device": str(device),
                "source_weights": source_weights,
                "phase_weights": phase_weights,
                "bootstrap_effective_variety": teacher_variety,
                "bootstrap_effective_time_budget_ms": teacher_time_budget_ms,
                "scenarios": scenarios,
                "board_channels": board_channels_for_view(args.input_view),
                "side_size": SIDE_SIZE,
                "start_iteration": start_iteration,
                "final_iteration": start_iteration + args.iterations,
                "history_plies": args.history_plies,
                "encoder_arch": getattr(model, "encoder_arch", None),
                "encoder_hidden_size": getattr(model, "encoder_hidden_size", None),
                "freeze_encoder": getattr(model, "freeze_encoder", None),
                "encoder_checkpoint": str(args.encoder_checkpoint) if args.encoder_checkpoint is not None else None,
                "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
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
        bootstrap_profile: Dict[str, float] = {}
        progress.phase(0, "heuristic_bootstrap", f"{args.heuristic_bootstrap_games} games")
        records = heuristic_bootstrap_records(
            engine,
            games=args.heuristic_bootstrap_games,
            max_plies=args.max_plies,
            seed=args.seed + 25_000,
            cap_value=args.cap_value,
            input_view=args.input_view,
            bootstrap_policy=args.bootstrap_policy,
            heuristic_variety=teacher_variety,
            heuristic_time_budget_ms=teacher_time_budget_ms,
            collect_metrics=False,
            history_plies=args.history_plies,
            profile=bootstrap_profile,
        )
        samples = [sample for record in records for sample in record.samples]
        logger.write(
            {
                "event": "generate",
                "phase": "heuristic_bootstrap",
                "iteration": 0,
                "seconds": round(time.perf_counter() - started_at, 3),
                "summary": generation_summary(records, samples),
                "profile": {key: round(value, 6) for key, value in sorted(bootstrap_profile.items())},
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

    final_iteration = start_iteration + args.iterations
    for iteration in range(start_iteration + 1, final_iteration + 1):
        iteration_samples: List[Sample] = []
        if args.heuristic_bootstrap_per_iteration > 0:
            started_at = time.perf_counter()
            bootstrap_profile = {}
            progress.phase(iteration, "heuristic_bootstrap_iteration", f"{args.heuristic_bootstrap_per_iteration} games")
            records = heuristic_bootstrap_records(
                engine,
                games=args.heuristic_bootstrap_per_iteration,
                max_plies=args.max_plies,
                seed=args.seed + 35_000 + iteration,
                cap_value=args.cap_value,
                input_view=args.input_view,
                bootstrap_policy=args.bootstrap_policy,
                heuristic_variety=teacher_variety,
                heuristic_time_budget_ms=teacher_time_budget_ms,
                collect_metrics=False,
                history_plies=args.history_plies,
                profile=bootstrap_profile,
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
                    "profile": {key: round(value, 6) for key, value in sorted(bootstrap_profile.items())},
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
                    kind_temperature=args.kind_temperature,
                    tuning_temperature=args.tuning_temperature,
                    force_first_tune_prob=args.force_first_tune_prob,
                    device=device,
                    cap_value=args.cap_value,
                    collect_metrics=False,
                    input_view=args.input_view,
                    full_policy=True,
                    max_tuning_actions=args.max_tuning_actions,
                    history_plies=args.history_plies,
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
            scenario_bootstrap_profile: Dict[str, float] = {}
            progress.phase(iteration, "scenario_heuristic_bootstrap", f"{args.scenario_bootstrap_per_iteration} games")
            records = heuristic_bootstrap_records(
                engine,
                games=args.scenario_bootstrap_per_iteration,
                max_plies=args.max_plies,
                seed=args.seed + 66_000 + iteration,
                cap_value=args.cap_value,
                input_view=args.input_view,
                bootstrap_policy=args.bootstrap_policy,
                heuristic_variety=teacher_variety,
                heuristic_time_budget_ms=teacher_time_budget_ms,
                initial_states=scenario_states,
                collect_metrics=False,
                history_plies=args.history_plies,
                profile=scenario_bootstrap_profile,
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
                    "profile": {key: round(value, 6) for key, value in sorted(scenario_bootstrap_profile.items())},
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
                    kind_temperature=args.kind_temperature,
                    tuning_temperature=args.tuning_temperature,
                    force_first_tune_prob=args.force_first_tune_prob,
                    device=device,
                    cap_value=args.cap_value,
                    collect_metrics=False,
                    input_view=args.input_view,
                    full_policy=True,
                    max_tuning_actions=args.max_tuning_actions,
                    initial_states=scenario_states,
                    history_plies=args.history_plies,
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
            run_tactical_eval_phase(engine, model, args, device, iteration, logger, progress, scenarios)
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
            save_checkpoint(model, optimizer, args, logger, iteration, final_iteration)

    if args.iterations == 0 or args.save_every <= 0 or final_iteration % args.save_every != 0:
        save_checkpoint(model, optimizer, args, logger, final_iteration, final_iteration)
    engine.close()


if __name__ == "__main__":
    main()
