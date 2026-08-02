from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .engine import RustEngine
from .encoding import TUNING_ACTION_SIZE
from .model import PolicyValueNet, masked_policy_logits
from .selfplay import (
    CapValueMode,
    PolicyMode,
    Sample,
    batched_model_selfplay_records,
    rust_random_training_samples,
    selfplay_records,
    session_model_selfplay_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Wave Field policy/value self-play model.")
    parser.add_argument("--games", type=int, default=8, help="Self-play games per iteration.")
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per generated batch.")
    parser.add_argument("--iterations", type=int, default=1, help="Generate/train cycles.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rollout-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--policy", choices=("random", "model"), default="random")
    parser.add_argument("--cap-value", choices=("zero", "material"), default="material")
    parser.add_argument("--python-selfplay", action="store_true", help="Use the slower Python random self-play path.")
    parser.add_argument("--legacy-model-selfplay", action="store_true", help="Use the older full-state JSON model rollout path.")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda.")
    parser.add_argument("--checkpoint", type=Path, default=Path("training/checkpoints/policy_value.pt"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--build-engine", action="store_true")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def samples_to_tensors(samples: List[Sample], device: torch.device) -> Dict[str, torch.Tensor]:
    tuning_masks = [
        sample.legal_tuning_mask
        if sample.legal_tuning_mask is not None
        else np.zeros((TUNING_ACTION_SIZE,), dtype=np.float32)
        for sample in samples
    ]
    return {
        "board": torch.tensor(np.stack([sample.board for sample in samples]), dtype=torch.float32, device=device),
        "side": torch.tensor(np.stack([sample.side for sample in samples]), dtype=torch.float32, device=device),
        "legal_mask": torch.tensor(np.stack([sample.legal_mask for sample in samples]), dtype=torch.float32, device=device),
        "actions": torch.tensor([sample.action_index for sample in samples], dtype=torch.long, device=device),
        "action_kinds": torch.tensor([sample.action_kind for sample in samples], dtype=torch.long, device=device),
        "legal_tuning_mask": torch.tensor(np.stack(tuning_masks), dtype=torch.float32, device=device),
        "tuning_actions": torch.tensor([sample.tuning_action_index for sample in samples], dtype=torch.long, device=device),
        "values": torch.tensor([sample.value for sample in samples], dtype=torch.float32, device=device),
    }


def train_epoch(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    tensors: Dict[str, torch.Tensor],
    batch_size: int,
    progress: Callable[[int, int, Dict[str, float]], None] | None = None,
) -> Dict[str, float]:
    model.train()
    sample_count = tensors["actions"].shape[0]
    order = torch.randperm(sample_count, device=tensors["actions"].device)
    totals = {"loss": 0.0, "kind": 0.0, "policy": 0.0, "tuning": 0.0, "value": 0.0}
    batch_count = (sample_count + batch_size - 1) // batch_size

    for batch_number, start in enumerate(range(0, sample_count, batch_size), start=1):
        batch = order[start:start + batch_size]
        kind_logits, move_logits, tuning_logits = model.full_policy(tensors["board"][batch], tensors["side"][batch])
        _legacy_logits, predicted_values = model(tensors["board"][batch], tensors["side"][batch])
        action_kinds = tensors["action_kinds"][batch]
        kind_loss = F.cross_entropy(kind_logits, action_kinds)

        move_rows = action_kinds == 0
        if bool(move_rows.any()):
            policy_loss = F.cross_entropy(
                masked_policy_logits(move_logits[move_rows], tensors["legal_mask"][batch][move_rows]),
                tensors["actions"][batch][move_rows],
            )
        else:
            policy_loss = move_logits.sum() * 0.0

        tuning_rows = action_kinds == 1
        if bool(tuning_rows.any()):
            tuning_loss = F.cross_entropy(
                masked_policy_logits(tuning_logits[tuning_rows], tensors["legal_tuning_mask"][batch][tuning_rows]),
                tensors["tuning_actions"][batch][tuning_rows],
            )
        else:
            tuning_loss = tuning_logits.sum() * 0.0
        value_loss = F.mse_loss(predicted_values, tensors["values"][batch])
        loss = kind_loss + policy_loss + tuning_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        weight = batch.shape[0] / sample_count
        totals["loss"] += loss.item() * weight
        totals["kind"] += kind_loss.item() * weight
        totals["policy"] += policy_loss.item() * weight
        totals["tuning"] += tuning_loss.item() * weight
        totals["value"] += value_loss.item() * weight
        if progress is not None:
            progress(batch_number, batch_count, totals)

    return totals


def load_checkpoint(path: Path, model: PolicyValueNet, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    if not path.exists():
        return {"iterations": 0, "samples": 0}
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    engine = RustEngine()
    if args.build_engine:
        engine.build_release()

    model = PolicyValueNet(hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    prior: Dict[str, Any] = {"iterations": 0, "samples": 0}
    if args.resume:
        prior = load_checkpoint(args.checkpoint, model, optimizer)

    total_samples = int(prior.get("samples", 0))
    start_iteration = int(prior.get("iterations", 0))
    policy: PolicyMode = args.policy
    cap_value: CapValueMode = args.cap_value

    for iteration in range(1, args.iterations + 1):
        if policy == "random" and not args.python_selfplay:
            samples, batch_summary = rust_random_training_samples(
                engine,
                games=args.games,
                max_plies=args.max_plies,
                seed=args.seed + start_iteration + iteration,
                cap_value=cap_value,
            )
            generated_games = int(batch_summary["games"])
            decisive = int(batch_summary["decisive"])
            generator_label = "rust-random"
        elif policy == "model":
            rollout = batched_model_selfplay_records if args.legacy_model_selfplay else session_model_selfplay_records
            records = rollout(
                engine,
                model,
                games=args.games,
                max_plies=args.max_plies,
                seed=args.seed + start_iteration + iteration,
                temperature=args.temperature,
                device=device,
                cap_value=cap_value,
                batch_size=args.rollout_batch_size,
            )
            samples = [sample for record in records for sample in record.samples]
            generated_games = len(records)
            decisive = sum(1 for record in records if record.stats.decisive)
            generator_label = "batched-model-legacy" if args.legacy_model_selfplay else "rust-session-model"
        else:
            records = selfplay_records(
                engine,
                games=args.games,
                max_plies=args.max_plies,
                seed=args.seed + start_iteration + iteration,
                policy=policy,
                model=model if policy == "model" else None,
                temperature=args.temperature,
                device=device,
                cap_value=cap_value,
                collect_metrics=False,
            )
            samples = [sample for record in records for sample in record.samples]
            generated_games = len(records)
            decisive = sum(1 for record in records if record.stats.decisive)
            generator_label = f"python-{policy}"
        if not samples:
            raise RuntimeError("No training samples generated")

        tensors = samples_to_tensors(samples, device)
        total_samples += len(samples)
        print(
            f"iteration={start_iteration + iteration} generator={generator_label} "
            f"generated_games={generated_games} samples={len(samples)} decisive={decisive} device={device}"
        )

        for epoch in range(1, args.epochs + 1):
            losses = train_epoch(model, optimizer, tensors, batch_size=args.batch_size)
            print(
                f"iteration={start_iteration + iteration} epoch={epoch} "
                f"loss={losses['loss']:.4f} kind={losses['kind']:.4f} "
                f"policy={losses['policy']:.4f} tuning={losses['tuning']:.4f} value={losses['value']:.4f}"
            )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "samples": total_samples,
            "iterations": start_iteration + args.iterations,
            "hidden_size": args.hidden_size,
            "policy": policy,
            "cap_value": cap_value,
        },
        args.checkpoint,
    )
    print(f"saved {args.checkpoint}")


if __name__ == "__main__":
    main()
