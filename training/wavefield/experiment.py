from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

from .augmentation import paired_samples
from .eval import aggregate
from .encoding import SIDE_SIZE, board_channels_for_view
from .engine import INITIAL_STATE_PATH, RustEngine, load_initial_state
from .model import PolicyValueNet
from .replay import ReplayBuffer
from .mcts import MctsConfig, mcts_selfplay_records, parallel_mcts_selfplay_records
from .scenarios import DEFAULT_SCENARIOS, build_scenario_states, scenario_names
from .selfplay import Sample, heuristic_bootstrap_records, rust_random_training_samples, session_model_selfplay_records
from .train import evaluate_loss, resolve_device, samples_to_tensors, train_epoch


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
    parser.add_argument(
        "--bootstrap-mix",
        default="",
        help="Comma-separated teacher mix, for example easy=2,heuristic=6,hard=8. Empty uses --bootstrap-policy.",
    )
    parser.add_argument("--bootstrap-variety", type=float, default=None)
    parser.add_argument("--bootstrap-time-budget-ms", type=int, default=None)
    parser.add_argument("--heuristic-bootstrap-games", type=int, default=0)
    parser.add_argument("--heuristic-bootstrap-per-iteration", type=int, default=0)
    parser.add_argument("--random-games-per-iteration", type=int, default=0)
    parser.add_argument("--model-games", type=int, default=100)
    parser.add_argument("--mcts-games-per-iteration", type=int, default=0)
    parser.add_argument("--mcts-workers", type=int, default=1, help="Parallel CPU self-play games per iteration.")
    parser.add_argument("--mcts-inference-device", choices=("local", "cpu", "mps", "cuda"), default="local", help="Shared inference backend; local keeps one CPU model per worker.")
    parser.add_argument("--mcts-inference-max-batch", type=int, default=32)
    parser.add_argument("--mcts-inference-wait-ms", type=float, default=2.0)
    parser.add_argument("--mcts-simulations", type=int, default=128)
    parser.add_argument("--mcts-top-k", type=int, default=0, help="Candidates per node; 0 keeps every legal action.")
    parser.add_argument("--mcts-min-tune-candidates", type=int, default=1)
    parser.add_argument("--mcts-min-move-candidates", type=int, default=1)
    parser.add_argument("--mcts-depth", type=int, default=6)
    parser.add_argument("--mcts-value-discount", type=float, default=1.0, help="Outcome discount per piece move; tuning is undiscounted.")
    parser.add_argument("--mcts-repetition-limit", type=int, choices=[0, 3], default=0,
                        help="End self-play as a draw on the third identical turn-boundary position; 0 disables.")
    parser.add_argument("--mcts-c-puct", type=float, default=1.5)
    parser.add_argument("--mcts-dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--mcts-exploration-fraction", type=float, default=0.25)
    parser.add_argument("--mcts-sample-first-plies", type=int, default=20)
    parser.add_argument(
        "--mcts-force-first-tune-prob",
        type=float,
        default=0.0,
        help="MCTS self-play probability of forcing the first eligible action in a turn to be tuning.",
    )
    parser.add_argument(
        "--mcts-force-tune-prob",
        type=float,
        default=0.0,
        help="MCTS self-play probability of forcing any eligible search action to be tuning.",
    )
    parser.add_argument("--scenario-games-per-iteration", type=int, default=0)
    parser.add_argument("--scenario-bootstrap-per-iteration", type=int, default=0)
    parser.add_argument("--scenario-eval-games", type=int, default=0)
    parser.add_argument("--tactical-eval-games", type=int, default=0)
    parser.add_argument("--tactical-eval-max-plies", type=int, default=1)
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--replay-capacity", type=int, default=0, help="Retained positions; 0 trains only on the latest batch.")
    parser.add_argument("--replay-sample-size", type=int, default=4096, help="Positions drawn per iteration before epoch passes.")
    parser.add_argument("--spatial-augmentation", action="store_true", help="Use all eight square symmetries with paired color augmentation (sparse rules only).")
    parser.add_argument("--color-augmentation", action="store_true", help="Pair original positions with rotated, color-swapped targets within the existing row budget.")
    parser.add_argument("--effective-batch-size", type=int, default=None, help="Positions per optimizer update, accumulated through microbatches.")
    parser.add_argument("--target-optimizer-steps", type=int, default=0, help="Absolute optimizer-update target; 0 uses iteration count.")
    parser.add_argument("--large-batch-start-step", type=int, default=0)
    parser.add_argument("--recovery-checkpoints", action="store_true", help="Rotate latest/previous recovery snapshots every iteration.")
    parser.add_argument("--milestone-every-updates", type=int, default=1000)
    parser.add_argument("--keep-milestones", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-games-per-batch", type=int, default=0,
                        help="Reserve this many complete MCTS games per batch outside training replay.")
    parser.add_argument("--validation-capacity", type=int, default=2048,
                        help="Maximum positions in the rolling held-out game buffer.")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.0,
        help="Hold out this fraction of each generated training batch for validation loss logging.",
    )
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
    parser.add_argument("--max-tuning-actions", type=int, default=20, help="Tuning actions per turn; -1 means unlimited.")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--skip-self-eval", action="store_true", help="Skip identical-network evaluation; use checkpoint opponents instead.")
    parser.add_argument("--checkpoint-eval-games", type=int, default=0, help="Games per side against a frozen earlier self.")
    parser.add_argument("--checkpoint-eval-temperature", type=float, default=0.3)
    parser.add_argument("--eval-games", type=int, default=25)
    parser.add_argument("--eval-max-plies", type=int, default=None, help="Override --max-plies for periodic eval.")
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-pressure", action="store_true")
    parser.add_argument("--baseline-eval-games", type=int, default=0, help="Run side-swapped model-vs-baseline matches during eval.")
    parser.add_argument("--baseline-eval-max-plies", type=int, default=None, help="Override ply cap for baseline eval matches.")
    parser.add_argument("--baseline-opponents", default="heuristic", help="Comma-separated policies: heuristic,hard,easy,random.")
    parser.add_argument("--record-eval-replays", action="store_true", help="Write baseline eval game traces to eval_replays.jsonl.")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--cap-value", choices=("zero", "material", "position"), default="material")
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
    def __init__(self, path: Path, progress: TerminalProgress | None = None, append: bool = False) -> None:
        self.path = path
        self.progress = progress
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not append:
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


def parse_bootstrap_mix(raw: str) -> Dict[str, int]:
    weights = parse_weights(raw)
    invalid = [policy for policy in weights if policy not in {"easy", "heuristic", "hard"}]
    if invalid:
        raise ValueError(f"Invalid bootstrap policy in --bootstrap-mix: {invalid}. Expected easy, heuristic, or hard.")
    if weights and sum(weights.values()) <= 0:
        raise ValueError("--bootstrap-mix must include at least one positive policy weight.")
    return weights


def allocate_bootstrap_mix(
    total_games: int,
    default_policy: str,
    mix_weights: Dict[str, int],
) -> List[Tuple[str, int]]:
    if total_games <= 0:
        return []
    if not mix_weights:
        return [(default_policy, total_games)]

    total_weight = sum(mix_weights.values())
    raw_counts = [
        (policy, total_games * weight / total_weight)
        for policy, weight in mix_weights.items()
        if weight > 0
    ]
    counts = {policy: int(raw + 0.5) for policy, raw in raw_counts}
    delta = total_games - sum(counts.values())
    if delta > 0:
        fractions = sorted(
            ((raw - int(raw), policy) for policy, raw in raw_counts),
            reverse=True,
        )
        for _, policy in fractions[:delta]:
            counts[policy] += 1
    elif delta < 0:
        fractions = sorted(
            ((counts[policy] - raw, policy) for policy, raw in raw_counts),
            reverse=True,
        )
        for _, policy in fractions:
            if delta == 0:
                break
            if counts[policy] > 0:
                counts[policy] -= 1
                delta += 1
    return [(policy, counts[policy]) for policy in mix_weights if counts.get(policy, 0) > 0]


def bootstrap_variety(args: argparse.Namespace) -> float:
    if args.bootstrap_variety is not None:
        return args.bootstrap_variety
    return 0.0 if args.bootstrap_policy == "hard" else 0.55


def bootstrap_time_budget_ms(args: argparse.Namespace) -> int:
    if args.bootstrap_time_budget_ms is not None:
        return args.bootstrap_time_budget_ms
    return 1_500 if args.bootstrap_policy == "hard" else 10


def teacher_variety(args: argparse.Namespace, policy: str) -> float:
    if args.bootstrap_variety is not None:
        return args.bootstrap_variety
    return 0.0 if policy == "hard" else 0.55


def teacher_time_budget_ms(args: argparse.Namespace, policy: str) -> int:
    if args.bootstrap_time_budget_ms is not None and policy == "hard":
        return args.bootstrap_time_budget_ms
    if args.bootstrap_time_budget_ms is not None and not args.bootstrap_mix:
        return args.bootstrap_time_budget_ms
    return 1_500 if policy == "hard" else 10


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
        "repetition_draws": sum(record.stats.termination_reason == 'threefold_repetition' for record in records),
        "plies_avoided_by_repetition": sum(record.stats.plies_avoided_by_repetition for record in records),
        "tactics": {
            "games": len(records),
            "decisive_games": sum(record.stats.decisive for record in records),
            "decisive_plies": sum(record.stats.plies for record in records if record.stats.decisive),
            "rescues": sum(record.stats.rescues for record in records),
            "rescue_opportunities": sum(record.stats.rescue_opportunities for record in records),
            "rescues_by_player": {player: sum(record.stats.rescues_by_player[player] for record in records) for player in ('red', 'blue')},
            "opportunities_by_player": {player: sum(record.stats.rescue_opportunities_by_player[player] for record in records) for player in ('red', 'blue')},
            "pieces_lost": {player: sum(record.stats.losses_by_player[player] for record in records) for player in ('red', 'blue')},
            "turns": sum(sum(record.stats.ai_turns_by_player.values()) for record in records),
            "tuning_turns": sum(sum(record.stats.tune_turns_by_player.values()) for record in records),
            "tuning_actions": sum(sum(record.stats.tune_actions_by_player.values()) for record in records),
            "net_tuning_changes": sum(sum(record.stats.effective_tune_changes_by_player.values()) for record in records),
        },
        "mean_plies": (
            sum(record.stats.plies for record in records) / len(records)
            if records
            else 0.0
        ),
        "wins": {
            winner: sum(1 for record in records if (record.stats.winner or ("repetition_draw" if record.stats.termination_reason == 'threefold_repetition' else "capped")) == winner)
            for winner in ("red", "blue", "capped", "repetition_draw")
        },
        "metadata": sample_metadata_summary(samples),
    }


def generate_bootstrap_mix(
    engine: RustEngine,
    args: argparse.Namespace,
    total_games: int,
    seed: int,
    phase: str,
    progress: TerminalProgress,
    iteration: int,
    bootstrap_mix: Dict[str, int],
    initial_states: List[Dict[str, Any]] | None = None,
) -> Tuple[List[Any], List[Sample], Dict[str, float], Dict[str, int]]:
    plan = allocate_bootstrap_mix(total_games, args.bootstrap_policy, bootstrap_mix)
    records: List[Any] = []
    samples: List[Sample] = []
    profile: Dict[str, float] = {}
    policy_counts: Dict[str, int] = {}
    state_offset = 0

    for policy_index, (policy, games) in enumerate(plan):
        policy_states = None
        if initial_states is not None:
            policy_states = initial_states[state_offset:state_offset + games]
        policy_profile: Dict[str, float] = {}
        detail = f"{games} {policy} games"
        progress.phase(iteration, phase, detail)
        policy_records = heuristic_bootstrap_records(
            engine,
            games=games,
            max_plies=args.max_plies,
            seed=seed + policy_index * 10_000,
            cap_value=args.cap_value,
            input_view=args.input_view,
            bootstrap_policy=policy,
            heuristic_variety=teacher_variety(args, policy),
            heuristic_time_budget_ms=teacher_time_budget_ms(args, policy),
            initial_states=policy_states,
            collect_metrics=False,
            history_plies=args.history_plies,
            profile=policy_profile,
        )
        policy_samples = [sample for record in policy_records for sample in record.samples]
        records.extend(policy_records)
        samples.extend(policy_samples)
        policy_counts[policy] = policy_counts.get(policy, 0) + games
        for key, value in policy_profile.items():
            profile[f"{policy}_{key}"] = profile.get(f"{policy}_{key}", 0.0) + value
        state_offset += games

    return records, samples, profile, policy_counts


def partition_validation_games(records: List[Any], count: int, iteration: int) -> Tuple[List[Sample], List[Sample]]:
    """Rotate held-out game slots deterministically, before replay insertion."""
    if count < 0 or (count and count >= len(records)):
        raise ValueError('Validation must leave at least one MCTS game for training')
    heldout = {(iteration + offset) % len(records) for offset in range(count)}
    training, validation = [], []
    for index, record in enumerate(records):
        (validation if index in heldout else training).extend(record.samples)
    return training, validation


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
    validation_fraction: float = 0.0,
    effective_batch_size: int | None = None,
    max_optimizer_steps: int | None = None,
    heldout_samples: List[Sample] | None = None,
    color_augmentation: bool = False,
    spatial_augmentation: bool = False,
) -> int:
    if not samples:
        logger.write(
            {
                "event": "train_skipped",
                "phase": phase,
                "iteration": iteration,
                "reason": "no_samples",
            }
        )
        return 0

    training_samples = samples
    if heldout_samples is not None and validation_fraction > 0:
        raise ValueError('Use held-out games or a per-batch validation fraction, not both')
    validation_samples: List[Sample] = heldout_samples or []
    if validation_fraction > 0.0 and len(samples) > 1:
        validation_count = max(1, int(round(len(samples) * validation_fraction)))
        validation_count = min(validation_count, len(samples) - 1)
        rng = np.random.default_rng(10_000_000 + iteration)
        order = rng.permutation(len(samples))
        validation_indexes = set(int(index) for index in order[:validation_count])
        validation_samples = [sample for index, sample in enumerate(samples) if index in validation_indexes]
        training_samples = [sample for index, sample in enumerate(samples) if index not in validation_indexes]

    original_training_count = len(training_samples)
    if spatial_augmentation and not color_augmentation:
        raise ValueError("Spatial augmentation requires paired color augmentation")
    if color_augmentation:
        training_samples = paired_samples(training_samples, seed=20_000_000 + iteration, spatial=spatial_augmentation)
    tensors = samples_to_tensors(training_samples, device)
    validation_tensors = samples_to_tensors(validation_samples, device) if validation_samples else None
    updates = 0
    for epoch in range(1, epochs + 1):
        if max_optimizer_steps is not None and updates >= max_optimizer_steps:
            break
        started_at = time.perf_counter()
        losses = train_epoch(
            model,
            optimizer,
            tensors,
            batch_size=batch_size,
            effective_batch_size=effective_batch_size,
            max_optimizer_steps=None if max_optimizer_steps is None else max_optimizer_steps - updates,
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
        updates += int(losses["optimizer_steps"])
        validation_losses = (
            evaluate_loss(model, validation_tensors, batch_size=batch_size)
            if validation_tensors is not None
            else None
        )
        progress.clear()
        event = {
            "event": "train_epoch",
            "phase": phase,
            "iteration": iteration,
            "epoch": epoch,
            "samples": int(losses["samples_processed"]),
            "microbatch_size": batch_size,
            "effective_batch_size": effective_batch_size or batch_size,
            "raw_samples": len(samples),
            "color_augmentation": color_augmentation,
            "spatial_augmentation": spatial_augmentation,
            "augmentation_variants": 16 if spatial_augmentation else (2 if color_augmentation else 1),
            "original_training_positions": len(training_samples) // 2 if color_augmentation else original_training_count,
            "validation_samples": len(validation_samples),
            "validation_source": "heldout_games" if heldout_samples is not None else "batch_split",
            "seconds": round(time.perf_counter() - started_at, 3),
            "loss": round(losses["loss"], 6),
            "kind": round(losses["kind"], 6),
            "policy": round(losses["policy"], 6),
            "tuning": round(losses["tuning"], 6),
            "value": round(losses["value"], 6),
            "history_plies": getattr(model, "history_plies", 1),
            "optimizer_steps": int(losses["optimizer_steps"]),
        }
        if validation_losses is not None:
            event.update(
                {
                    "validation_loss": round(validation_losses["loss"], 6),
                    "validation_kind": round(validation_losses["kind"], 6),
                    "validation_policy": round(validation_losses["policy"], 6),
                    "validation_tuning": round(validation_losses["tuning"], 6),
                    "validation_value": round(validation_losses["value"], 6),
                }
            )
        logger.write(event)
    return updates


def run_checkpoint_eval(
    engine: RustEngine,
    model: PolicyValueNet,
    opponent: PolicyValueNet,
    args: argparse.Namespace,
    device: torch.device,
    iteration: int,
    opponent_iteration: int,
    logger: JsonlLogger,
) -> None:
    from .match import play_match_game

    started = time.perf_counter()
    wins = losses = capped = 0
    # Evaluation randomness must not change the learner's training RNG stream.
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        for model_side in ("blue", "red"):
            other_side = "red" if model_side == "blue" else "blue"
            for game in range(args.checkpoint_eval_games):
                seed = args.seed + 200_000 + iteration * 100 + game
                torch.manual_seed(seed)
                record = play_match_game(
                    engine, policies={"blue": "model", "red": "model"}, model=model,
                    models_by_player={model_side: model, other_side: opponent},
                    device=device, max_plies=args.eval_max_plies or args.max_plies,
                    seed=seed, temperature=args.checkpoint_eval_temperature,
                    input_view=args.input_view, full_policy=True,
                    max_tuning_actions=args.max_tuning_actions, collect_metrics=False,
                )
                if record.stats.winner == model_side:
                    wins += 1
                elif record.stats.winner == other_side:
                    losses += 1
                else:
                    capped += 1
    games = wins + losses + capped
    logger.write({
        "event": "checkpoint_eval", "iteration": iteration,
        "opponent_iteration": opponent_iteration, "policy": "network_without_search",
        "temperature": args.checkpoint_eval_temperature,
        "games": games, "wins": wins, "losses": losses, "capped": capped,
        "score": (wins + 0.5 * capped) / games,
        "seconds": round(time.perf_counter() - started, 3),
    })


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
                    record_replay=args.record_eval_replays,
                )
                for game in range(games_per_side)
            ]
            summary = aggregate(records)
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
                    "summary": summary,
                }
            )
            if args.record_eval_replays:
                replay_path = args.run_dir / "eval_replays.jsonl"
                with replay_path.open("a", encoding="utf-8") as handle:
                    for game, record in enumerate(records):
                        handle.write(
                            json.dumps(
                                {
                                    "event": "baseline_eval_replay",
                                    "iteration": iteration,
                                    "game": game,
                                    "opponent": opponent,
                                    "model_side": model_side,
                                    "policies": policies,
                                    "max_plies": max_plies,
                                    "stats": record.stats.to_dict(),
                                    "replay": record.replay,
                                },
                                sort_keys=True,
                            )
                            + "\n"
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
    replay: ReplayBuffer | None = None,
    optimizer_steps: int = 0,
    checkpoint_opponent: PolicyValueNet | None = None,
    opponent_iteration: int = 0,
    validation_replay: ReplayBuffer | None = None,
) -> None:
    checkpoint_path = args.run_dir / ("checkpoint.pt" if iteration == final_iteration else f"checkpoint-iter-{iteration}.pt")
    if args.recovery_checkpoints:
        checkpoint_path = args.run_dir / "checkpoint-latest.pt"
    temporary_path = checkpoint_path.with_suffix(".tmp")
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
            "rules_variant": load_initial_state().get("variant", "classic"),
            "replay": replay.state_dict() if replay is not None else None,
            "validation_replay": validation_replay.state_dict() if validation_replay is not None else None,
            "optimizer_steps": optimizer_steps,
            "config": serializable_config(args),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "checkpoint_opponent": checkpoint_opponent.state_dict() if checkpoint_opponent is not None else None,
            "opponent_iteration": opponent_iteration,
        },
        temporary_path,
    )
    if args.recovery_checkpoints:
        with temporary_path.open("rb") as saved:
            os.fsync(saved.fileno())
    if args.recovery_checkpoints and checkpoint_path.exists():
        previous = args.run_dir / "checkpoint-previous.pt"
        link = previous.with_suffix(".tmp")
        link.unlink(missing_ok=True)
        os.link(checkpoint_path, link)
        link.replace(previous)
    temporary_path.replace(checkpoint_path)
    if args.recovery_checkpoints:
        metadata = {"iteration": iteration, "optimizer_steps": optimizer_steps,
                    "large_batch_updates": max(0, optimizer_steps - args.large_batch_start_step),
                    "checkpoint": str(checkpoint_path)}
        manifest = args.run_dir / "checkpoint-latest.json"
        temp = manifest.with_suffix(".tmp")
        temp.write_text(json.dumps(metadata) + "\n")
        temp.replace(manifest)
        existing = sorted(args.run_dir.glob("checkpoint-update-*.pt"))
        latest_step = max((int(path.stem.split("-")[-1]) for path in existing), default=0)
        effective_updates = metadata["large_batch_updates"]
        if args.milestone_every_updates > 0 and effective_updates - latest_step >= args.milestone_every_updates:
            os.link(checkpoint_path, args.run_dir / f"checkpoint-update-{effective_updates:09d}.pt")
            existing = sorted(args.run_dir.glob("checkpoint-update-*.pt"))
            for old in existing[:-args.keep_milestones]:
                old.unlink()
    logger.write({"event": "saved", "iteration": iteration, "checkpoint": str(checkpoint_path)})


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    source_weights = parse_weights(args.source_weights)
    phase_weights = parse_weights(args.phase_weights)
    bootstrap_mix = parse_bootstrap_mix(args.bootstrap_mix)
    scenarios = scenario_names(args.scenarios)
    effective_teacher_variety = bootstrap_variety(args)
    effective_teacher_time_budget_ms = bootstrap_time_budget_ms(args)
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
    if args.cap_value == "position" and (args.pretrain_random_games > 0 or args.random_games_per_iteration > 0):
        raise ValueError("--cap-value position requires Python self-play; Rust random batches do not support positional adjudication.")
    if args.encoder_checkpoint is not None and args.model_arch != "encoder_sequence":
        raise ValueError("--encoder-checkpoint requires --model-arch encoder_sequence.")
    if args.model_arch == "encoder_sequence" and args.encoder_checkpoint is None and not args.unfreeze_encoder:
        raise ValueError("encoder_sequence needs --encoder-checkpoint unless --unfreeze-encoder trains it from scratch.")
    if not 0.0 <= args.force_first_tune_prob <= 1.0:
        raise ValueError("--force-first-tune-prob must be between 0 and 1.")
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be at least 0 and less than 1.")
    if args.validation_games_per_batch < 0 or args.validation_capacity < 1:
        raise ValueError('Validation game count must be nonnegative and capacity positive')
    if args.validation_games_per_batch and (args.validation_games_per_batch >= args.mcts_games_per_iteration or args.validation_fraction):
        raise ValueError('Held-out games require remaining MCTS training games and validation-fraction 0')
    if not 0.0 <= args.mcts_force_first_tune_prob <= 1.0:
        raise ValueError("--mcts-force-first-tune-prob must be between 0 and 1.")
    if not 0.0 <= args.mcts_force_tune_prob <= 1.0:
        raise ValueError("--mcts-force-tune-prob must be between 0 and 1.")
    if args.mcts_dirichlet_alpha < 0.0:
        raise ValueError("--mcts-dirichlet-alpha must be nonnegative.")
    if not 0.0 <= args.mcts_exploration_fraction <= 1.0:
        raise ValueError("--mcts-exploration-fraction must be between 0 and 1.")
    if args.mcts_sample_first_plies < 0:
        raise ValueError("--mcts-sample-first-plies must be nonnegative.")
    if not 0 < args.mcts_value_discount <= 1:
        raise ValueError("--mcts-value-discount must be in (0, 1].")
    if args.mcts_inference_max_batch < 1 or args.mcts_inference_wait_ms < 0:
        raise ValueError("Inference batch size must be positive and wait time nonnegative.")
    if args.mcts_workers < 1:
        raise ValueError("--mcts-workers must be at least 1.")
    if args.mcts_workers > 1 and device.type != "cpu":
        raise ValueError("--mcts-workers > 1 currently requires --device cpu.")
    if args.replay_capacity < 0 or args.replay_sample_size < 1:
        raise ValueError("Replay capacity must be nonnegative and replay sample size positive.")
    if args.checkpoint_eval_games < 0:
        raise ValueError("--checkpoint-eval-games must be nonnegative.")
    if args.checkpoint_eval_temperature < 0 or args.mcts_top_k < 0:
        raise ValueError("Evaluation temperature and MCTS candidate limit must be nonnegative.")
    if args.effective_batch_size is not None and args.effective_batch_size < args.batch_size:
        raise ValueError("Effective batch size must be at least the microbatch size.")
    if args.target_optimizer_steps < 0 or args.keep_milestones < 1:
        raise ValueError("Update target must be nonnegative and milestone retention positive.")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epoch count and batch size must be positive.")
    if args.max_tuning_actions < -1:
        raise ValueError("--max-tuning-actions must be -1 (unlimited) or nonnegative.")
    if args.resume_checkpoint is not None and (args.pretrain_random_games > 0 or args.heuristic_bootstrap_games > 0):
        raise ValueError("Resume runs should use per-iteration generation, not bootstrap phases.")
    torch.manual_seed(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    training_lock = (args.run_dir / ".training.lock").open("w")
    if os.name == "posix":
        import fcntl
        fcntl.flock(training_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start_iteration = 0
    progress = TerminalProgress(args.progress, args.iterations)
    logger = JsonlLogger(args.run_dir / "events.jsonl", progress, append=args.resume_checkpoint is not None)

    if args.spatial_augmentation and (not args.color_augmentation or load_initial_state()["variant"] != "sparse"):
        raise ValueError("Spatial augmentation requires sparse rules and --color-augmentation")
    engine = RustEngine(retain_history=False)
    replay = ReplayBuffer(args.replay_capacity) if args.replay_capacity else None
    validation_replay = ReplayBuffer(args.validation_capacity) if args.validation_games_per_batch else None
    optimizer_steps = 0
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
        if checkpoint.get("rules_variant", "classic") != load_initial_state().get("variant", "classic"):
            raise ValueError("Checkpoint rules differ from the current game's rules; start a fresh run.")
        model.load_state_dict(checkpoint["model"], strict=False)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_iteration = int(checkpoint.get("iteration", 0))
        optimizer_steps = int(checkpoint.get("optimizer_steps", 0))
        if replay is not None and checkpoint.get("replay") is not None:
            replay.load_state_dict(checkpoint["replay"])
        if validation_replay is not None and checkpoint.get('validation_replay') is not None:
            validation_replay.load_state_dict(checkpoint['validation_replay'])
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if checkpoint.get("numpy_rng_state") is not None:
            np.random.set_state(checkpoint["numpy_rng_state"])
        progress.iterations = start_iteration + args.iterations

    checkpoint_opponent = copy.deepcopy(model).eval() if args.checkpoint_eval_games else None
    opponent_iteration = start_iteration
    if args.resume_checkpoint is not None and checkpoint_opponent is not None and checkpoint.get("checkpoint_opponent") is not None:
        checkpoint_opponent.load_state_dict(checkpoint["checkpoint_opponent"])
        opponent_iteration = int(checkpoint.get("opponent_iteration", start_iteration))

    logger.write(
        {
            "event": "config",
            "config": {
                **serializable_config(args),
                "run_dir": str(args.run_dir),
                "device": str(device),
                "rules_variant": load_initial_state().get("variant", "classic"),
                "initial_state_path": str(INITIAL_STATE_PATH),
                "source_weights": source_weights,
                "phase_weights": phase_weights,
                "bootstrap_mix": bootstrap_mix,
                "bootstrap_effective_variety": effective_teacher_variety,
                "bootstrap_effective_time_budget_ms": effective_teacher_time_budget_ms,
                "scenarios": scenarios,
                "board_channels": board_channels_for_view(args.input_view),
                "side_size": SIDE_SIZE,
                "start_iteration": start_iteration,
                "optimizer_steps_start": optimizer_steps,
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
        optimizer_steps += train_samples(
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
            validation_fraction=args.validation_fraction,
        )

    if args.heuristic_bootstrap_games > 0:
        started_at = time.perf_counter()
        records, samples, bootstrap_profile, policy_counts = generate_bootstrap_mix(
            engine,
            args,
            args.heuristic_bootstrap_games,
            seed=args.seed + 25_000,
            phase="heuristic_bootstrap",
            progress=progress,
            iteration=0,
            bootstrap_mix=bootstrap_mix,
        )
        logger.write(
            {
                "event": "generate",
                "phase": "heuristic_bootstrap",
                "iteration": 0,
                "seconds": round(time.perf_counter() - started_at, 3),
                "summary": generation_summary(records, samples),
                "policy_counts": policy_counts,
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
        optimizer_steps += train_samples(
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
            validation_fraction=args.validation_fraction,
        )

    final_iteration = start_iteration + args.iterations
    for iteration in range(start_iteration + 1, final_iteration + 1):
        if args.target_optimizer_steps and optimizer_steps >= args.target_optimizer_steps:
            break
        iteration_samples: List[Sample] = []
        if args.heuristic_bootstrap_per_iteration > 0:
            started_at = time.perf_counter()
            records, bootstrap_samples, bootstrap_profile, policy_counts = generate_bootstrap_mix(
                engine,
                args,
                args.heuristic_bootstrap_per_iteration,
                seed=args.seed + 35_000 + iteration,
                phase="heuristic_bootstrap_iteration",
                progress=progress,
                iteration=iteration,
                bootstrap_mix=bootstrap_mix,
            )
            iteration_samples.extend(bootstrap_samples)
            logger.write(
                {
                    "event": "generate",
                    "phase": "heuristic_bootstrap_iteration",
                    "iteration": iteration,
                    "seconds": round(time.perf_counter() - started_at, 3),
                    "summary": generation_summary(records, bootstrap_samples),
                    "policy_counts": policy_counts,
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

        if args.mcts_games_per_iteration > 0:
            started_at = time.perf_counter()
            progress.phase(iteration, "mcts_model_session", f"{args.mcts_games_per_iteration} games")
            def record_mcts_game(index: int, record: Any) -> None:
                logger.write({
                    "event": "selfplay_game", "iteration": iteration,
                    "game": index + 1, "games": args.mcts_games_per_iteration,
                    "plies": record.stats.plies, "winner": record.stats.winner,
                    "capped": record.final_state["status"] == "playing" and not record.stats.termination_reason,
                    "termination_reason": record.stats.termination_reason,
                    "plies_avoided_by_repetition": record.stats.plies_avoided_by_repetition,
                    "samples": len(record.samples),
                    "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                })

            mcts_config = MctsConfig(
                simulations=args.mcts_simulations,
                repetition_limit=args.mcts_repetition_limit,
                value_discount=args.mcts_value_discount,
                c_puct=args.mcts_c_puct,
                top_k=args.mcts_top_k,
                min_tune_candidates=args.mcts_min_tune_candidates,
                min_move_candidates=args.mcts_min_move_candidates,
                max_depth=args.mcts_depth,
                max_tuning_actions=args.max_tuning_actions,
                policy_temperature=args.temperature,
                dirichlet_alpha=args.mcts_dirichlet_alpha,
                exploration_fraction=args.mcts_exploration_fraction,
                sample_first_plies=args.mcts_sample_first_plies,
                force_first_tune_prob=args.mcts_force_first_tune_prob,
                force_tune_prob=args.mcts_force_tune_prob,
            )
            inference_stats: Dict[str, Any] = {}
            if args.mcts_workers > 1 or args.mcts_inference_device != "local":
                records = parallel_mcts_selfplay_records(
                    model,
                    games=args.mcts_games_per_iteration,
                    workers=args.mcts_workers,
                    inference_device=None if args.mcts_inference_device == "local" else args.mcts_inference_device,
                    inference_max_batch=args.mcts_inference_max_batch,
                    inference_wait_ms=args.mcts_inference_wait_ms,
                    inference_stats=inference_stats,
                    max_plies=args.max_plies,
                    seed=args.seed + 60_000 + iteration,
                    config=mcts_config,
                    input_view=args.input_view,
                    history_plies=args.history_plies,
                    cap_value=args.cap_value,
                    on_game_complete=record_mcts_game,
                )
            else:
                records = mcts_selfplay_records(
                    engine,
                    model,
                    games=args.mcts_games_per_iteration,
                    max_plies=args.max_plies,
                    seed=args.seed + 60_000 + iteration,
                    config=mcts_config,
                    device=device,
                    input_view=args.input_view,
                    history_plies=args.history_plies,
                    cap_value=args.cap_value,
                    on_game_complete=record_mcts_game,
                )
            mcts_samples = [sample for record in records for sample in record.samples]
            if validation_replay is not None:
                train_game_samples, heldout_game_samples = partition_validation_games(records, args.validation_games_per_batch, iteration)
                validation_replay.extend(heldout_game_samples)
                iteration_samples.extend(train_game_samples)
                logger.write({'event': 'validation_split', 'iteration': iteration,
                              'heldout_games': args.validation_games_per_batch,
                              'new_validation_samples': len(heldout_game_samples),
                              'validation_samples': len(validation_replay),
                              'validation_total_seen': validation_replay.total_seen,
                              'training_game_samples': len(train_game_samples)})
            else:
                iteration_samples.extend(mcts_samples)
            logger.write(
                {
                    "event": "generate",
                    "phase": "mcts_model_session",
                    "search_simulations": len(mcts_samples) * args.mcts_simulations,
                    "inference": inference_stats,
                    "iteration": iteration,
                    "seconds": round(time.perf_counter() - started_at, 3),
                    "summary": generation_summary(records, mcts_samples),
                    "mcts": {
                        "simulations": args.mcts_simulations,
                        "workers": args.mcts_workers,
                        "top_k": args.mcts_top_k,
                        "min_tune_candidates": args.mcts_min_tune_candidates,
                        "min_move_candidates": args.mcts_min_move_candidates,
                        "depth": args.mcts_depth,
                        "c_puct": args.mcts_c_puct,
                        "dirichlet_alpha": args.mcts_dirichlet_alpha,
                        "exploration_fraction": args.mcts_exploration_fraction,
                        "sample_first_plies": args.mcts_sample_first_plies,
                        "force_first_tune_prob": args.mcts_force_first_tune_prob,
                        "force_tune_prob": args.mcts_force_tune_prob,
                    },
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
            records, bootstrap_samples, scenario_bootstrap_profile, policy_counts = generate_bootstrap_mix(
                engine,
                args,
                args.scenario_bootstrap_per_iteration,
                seed=args.seed + 66_000 + iteration,
                phase="scenario_heuristic_bootstrap",
                progress=progress,
                iteration=iteration,
                bootstrap_mix=bootstrap_mix,
                initial_states=scenario_states,
            )
            iteration_samples.extend(bootstrap_samples)
            logger.write(
                {
                    "event": "generate",
                    "phase": "scenario_heuristic_bootstrap",
                    "iteration": iteration,
                    "seconds": round(time.perf_counter() - started_at, 3),
                    "summary": generation_summary(records, bootstrap_samples),
                    "policy_counts": policy_counts,
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

        if replay is not None:
            replay.extend(iteration_samples)
            update_samples = replay.draw(args.replay_sample_size, seed=args.seed + 300_000 + iteration)
        else:
            update_samples = iteration_samples
        training_samples = replay_weight_samples(update_samples, source_weights, phase_weights)
        logger.write(
            {
                "event": "train_batch",
                "phase": "curriculum",
                "iteration": iteration,
                "raw_samples": len(iteration_samples),
                "training_samples": len(training_samples),
                "replay_size": len(replay) if replay is not None else 0,
                "replay_total_seen": replay.total_seen if replay is not None else len(iteration_samples),
                "summary": sample_metadata_summary(training_samples),
            }
        )
        optimizer_steps += train_samples(
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
            validation_fraction=args.validation_fraction,
            effective_batch_size=args.effective_batch_size,
            max_optimizer_steps=None if not args.target_optimizer_steps else args.target_optimizer_steps - optimizer_steps,
            heldout_samples=list(validation_replay.samples) if validation_replay is not None else None,
            color_augmentation=args.color_augmentation,
            spatial_augmentation=args.spatial_augmentation,
        )
        logger.write({"event": "learning_update", "iteration": iteration, "optimizer_steps_total": optimizer_steps})
        if args.eval_every > 0 and iteration % args.eval_every == 0:
            if not args.skip_self_eval:
                run_session_eval(engine, model, args, device, iteration, logger, progress)
            if checkpoint_opponent is not None:
                run_checkpoint_eval(engine, model, checkpoint_opponent, args, device, iteration, opponent_iteration, logger)
                checkpoint_opponent.load_state_dict(model.state_dict())
                opponent_iteration = iteration
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
        if args.recovery_checkpoints or (args.save_every > 0 and iteration % args.save_every == 0):
            save_checkpoint(model, optimizer, args, logger, iteration, final_iteration, replay, optimizer_steps, checkpoint_opponent, opponent_iteration, validation_replay)
        if args.target_optimizer_steps and optimizer_steps >= args.target_optimizer_steps:
            logger.write({"event": "target_complete", "iteration": iteration, "optimizer_steps_total": optimizer_steps})
            break

    if not args.recovery_checkpoints and (args.iterations == 0 or args.save_every <= 0 or final_iteration % args.save_every != 0):
        save_checkpoint(model, optimizer, args, logger, final_iteration, final_iteration, replay, optimizer_steps, checkpoint_opponent, opponent_iteration, validation_replay)
    engine.close()


if __name__ == "__main__":
    main()
