from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .engine import RustEngine
from .model import PolicyValueNet, masked_policy_logits
from .selfplay import CapValueMode, PolicyMode, Sample, selfplay_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Wave Field policy/value self-play model.")
    parser.add_argument("--games", type=int, default=8, help="Self-play games per iteration.")
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per generated batch.")
    parser.add_argument("--iterations", type=int, default=1, help="Generate/train cycles.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--policy", choices=("random", "model"), default="random")
    parser.add_argument("--cap-value", choices=("zero", "material"), default="material")
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
    return {
        "board": torch.tensor(np.stack([sample.board for sample in samples]), dtype=torch.float32, device=device),
        "side": torch.tensor(np.stack([sample.side for sample in samples]), dtype=torch.float32, device=device),
        "legal_mask": torch.tensor(np.stack([sample.legal_mask for sample in samples]), dtype=torch.float32, device=device),
        "actions": torch.tensor([sample.action_index for sample in samples], dtype=torch.long, device=device),
        "values": torch.tensor([sample.value for sample in samples], dtype=torch.float32, device=device),
    }


def train_epoch(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    tensors: Dict[str, torch.Tensor],
    batch_size: int,
) -> Dict[str, float]:
    model.train()
    sample_count = tensors["actions"].shape[0]
    order = torch.randperm(sample_count, device=tensors["actions"].device)
    totals = {"loss": 0.0, "policy": 0.0, "value": 0.0}

    for start in range(0, sample_count, batch_size):
        batch = order[start:start + batch_size]
        logits, predicted_values = model(tensors["board"][batch], tensors["side"][batch])
        policy_loss = F.cross_entropy(masked_policy_logits(logits, tensors["legal_mask"][batch]), tensors["actions"][batch])
        value_loss = F.mse_loss(predicted_values, tensors["values"][batch])
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        weight = batch.shape[0] / sample_count
        totals["loss"] += loss.item() * weight
        totals["policy"] += policy_loss.item() * weight
        totals["value"] += value_loss.item() * weight

    return totals


def load_checkpoint(path: Path, model: PolicyValueNet, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    if not path.exists():
        return {"iterations": 0, "samples": 0}
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
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
        if not samples:
            raise RuntimeError("No training samples generated")

        tensors = samples_to_tensors(samples, device)
        total_samples += len(samples)
        decisive = sum(1 for record in records if record.stats.decisive)
        print(
            f"iteration={start_iteration + iteration} generated_games={len(records)} "
            f"samples={len(samples)} decisive={decisive} device={device}"
        )

        for epoch in range(1, args.epochs + 1):
            losses = train_epoch(model, optimizer, tensors, batch_size=args.batch_size)
            print(
                f"iteration={start_iteration + iteration} epoch={epoch} "
                f"loss={losses['loss']:.4f} policy={losses['policy']:.4f} value={losses['value']:.4f}"
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
