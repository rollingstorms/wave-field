from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .engine import RustEngine
from .model import PolicyValueNet, masked_policy_logits
from .selfplay import random_selfplay_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the first Wave Field policy/value smoke model.")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--checkpoint", type=Path, default=Path("training/checkpoints/policy_value.pt"))
    parser.add_argument("--build-engine", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    engine = RustEngine()
    if args.build_engine:
        engine.build_release()

    samples = random_selfplay_samples(engine, games=args.games, max_plies=args.max_plies, seed=args.seed)
    if not samples:
        raise RuntimeError("No training samples generated")

    board = torch.tensor(np.stack([sample.board for sample in samples]), dtype=torch.float32)
    side = torch.tensor(np.stack([sample.side for sample in samples]), dtype=torch.float32)
    legal_mask = torch.tensor(np.stack([sample.legal_mask for sample in samples]), dtype=torch.float32)
    actions = torch.tensor([sample.action_index for sample in samples], dtype=torch.long)
    values = torch.tensor([sample.value for sample in samples], dtype=torch.float32)

    model = PolicyValueNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        logits, predicted_values = model(board, side)
        policy_loss = F.cross_entropy(masked_policy_logits(logits, legal_mask), actions)
        value_loss = F.mse_loss(predicted_values, values)
        loss = policy_loss + value_loss
        loss.backward()
        optimizer.step()
        print(
            f"epoch={epoch} samples={len(samples)} "
            f"loss={loss.item():.4f} policy={policy_loss.item():.4f} value={value_loss.item():.4f}"
        )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "samples": len(samples)}, args.checkpoint)
    print(f"saved {args.checkpoint}")


if __name__ == "__main__":
    main()
