from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Tuple

import numpy as np
import torch
from torch import nn

from .encoding import InputView, encode_state
from .engine import RustEngine, load_initial_state
from .eval import load_model
from .model import PolicyValueNet, masked_policy_logits
from .selfplay import _no_move_loss
from .train import resolve_device


ProbeTask = Literal["binary", "multiclass", "regression"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Wave Field model activations and linear probes.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument(
        "--model-arch",
        choices=("conv", "residual", "transformer", "sequence_transformer", "encoder_sequence"),
        default=None,
    )
    parser.add_argument("--input-view", choices=("base", "piece_identity"), default=None)
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--max-plies", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--probe-epochs", type=int, default=120)
    parser.add_argument("--probe-lr", type=float, default=5.0e-3)
    parser.add_argument("--probe-batch-size", type=int, default=64)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def effective_rank(features: torch.Tensor, center: bool = True) -> Dict[str, float | int]:
    matrix = features.detach().float().cpu()
    if matrix.ndim > 2:
        matrix = matrix.flatten(start_dim=1)
    if center:
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
    if matrix.numel() == 0 or matrix.shape[0] < 2:
        return {"samples": int(matrix.shape[0]), "dimensions": int(matrix.shape[1] if matrix.ndim == 2 else 0)}

    singular_values = torch.linalg.svdvals(matrix)
    singular_values = singular_values[singular_values > 1.0e-8]
    if singular_values.numel() == 0:
        return {
            "samples": int(matrix.shape[0]),
            "dimensions": int(matrix.shape[1]),
            "effective_rank": 0.0,
            "participation_rank": 0.0,
            "stable_rank": 0.0,
            "top1_energy_share": 0.0,
            "top5_energy_share": 0.0,
        }

    energy = singular_values.square()
    energy_share = energy / energy.sum()
    entropy = -(energy_share * torch.log(energy_share.clamp_min(1.0e-12))).sum()
    participation = 1.0 / energy_share.square().sum()
    stable = energy.sum() / energy.max()
    top5 = energy_share[: min(5, energy_share.numel())].sum()
    return {
        "samples": int(matrix.shape[0]),
        "dimensions": int(matrix.shape[1]),
        "rank": int(singular_values.numel()),
        "effective_rank": round(float(torch.exp(entropy).item()), 3),
        "participation_rank": round(float(participation.item()), 3),
        "stable_rank": round(float(stable.item()), 3),
        "top1_energy_share": round(float(energy_share[0].item()), 4),
        "top5_energy_share": round(float(top5.item()), 4),
        "singular_value_ratio_min_max": round(float((singular_values[-1] / singular_values[0]).item()), 6),
    }


def _phase_for_ply(ply: int) -> int:
    if ply < 32:
        return 0
    if ply < 96:
        return 1
    return 2


def _piece_counts(state: Dict[str, Any]) -> Dict[str, int]:
    counts = {"red": 0, "blue": 0}
    for piece in state["pieces"]:
        counts[piece["owner"]] += 1
    return counts


def collect_states(
    engine: RustEngine,
    samples: int,
    max_plies: int,
    seed: int,
) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, float | int]]]:
    rng = random.Random(seed)
    collected: List[Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, float | int]]] = []
    game = 0
    while len(collected) < samples:
        state = load_initial_state()
        for ply in range(max_plies):
            if state["status"] != "playing" or len(collected) >= samples:
                break
            actions = engine.legal_actions(state)
            counts = _piece_counts(state)
            current = state["currentPlayer"]
            opponent = "blue" if current == "red" else "red"
            red_check = int(engine.king_unprotected(state, "red"))
            blue_check = int(engine.king_unprotected(state, "blue"))
            labels: Dict[str, float | int] = {
                "current_is_red": int(current == "red"),
                "current_king_unprotected": red_check if current == "red" else blue_check,
                "opponent_king_unprotected": blue_check if current == "red" else red_check,
                "red_king_unprotected": red_check,
                "blue_king_unprotected": blue_check,
                "has_legal_moves": int(bool(actions)),
                "legal_move_count": len(actions),
                "low_legal_count": int(len(actions) <= 8),
                "material_balance_red": counts["red"] - counts["blue"],
                "material_red_positive": int(counts["red"] > counts["blue"]),
                "total_pieces": counts["red"] + counts["blue"],
                "low_material": int(counts["red"] + counts["blue"] <= 8),
                "phase": _phase_for_ply(ply),
                "ply": ply,
            }
            collected.append((state, actions, labels))
            if not actions:
                state = _no_move_loss(state)
                break
            state = engine.apply_action(state, rng.choice(actions), analyze_checkmate=False)
        game += 1
        if game > samples * 2 and not collected:
            raise RuntimeError("failed to collect any reachable states")
    return collected


def capture_activations(
    model: PolicyValueNet,
    board: torch.Tensor,
    side: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    model.eval()
    activations: Dict[str, torch.Tensor] = {
        "board_flat": board.flatten(start_dim=1),
        "side": side,
    }
    with torch.no_grad():
        if model.architecture == "transformer":
            squares = board.permute(0, 2, 3, 1).reshape(board.shape[0], 49, model.board_channels)
            square_tokens = model.square_projection(squares)
            side_token = model.side_token(side).unsqueeze(1)
            tokens = torch.cat([side_token, square_tokens], dim=1) + model.position_embedding
            activations["tokens_pre_transformer"] = tokens.flatten(start_dim=1)
            encoded = tokens
            assert model.transformer is not None
            for index, layer in enumerate(model.transformer.layers):
                encoded = layer(encoded)
                activations[f"transformer_layer_{index}"] = encoded.flatten(start_dim=1)
            if model.transformer.norm is not None:
                encoded = model.transformer.norm(encoded)
                activations["transformer_norm"] = encoded.flatten(start_dim=1)
            embedding = encoded.flatten(start_dim=1)
        else:
            board_features = model.board(board)
            side_features = model.side(side)
            activations["board_features"] = board_features
            activations["side_features"] = side_features
            embedding = torch.cat([board_features, side_features], dim=1)
        activations["trunk_input"] = embedding
        hidden = model.trunk(embedding)
        activations["trunk_hidden"] = hidden
        kind_logits, move_logits, tune_logits = model.full_policy(board, side)
        activations["kind_logits"] = kind_logits
        activations["move_logits"] = move_logits
        activations["tune_logits"] = tune_logits
        activations["value"] = model.value(hidden)
    return activations


def legal_policy_summary(move_logits: torch.Tensor, masks: torch.Tensor) -> Dict[str, float]:
    masked = masked_policy_logits(move_logits, masks)
    probs = torch.softmax(masked, dim=1)
    legal_counts = masks.sum(dim=1)
    entropy = -(probs * torch.log(probs.clamp_min(1.0e-12))).sum(dim=1)
    return {
        "mean_legal_moves": round(float(legal_counts.mean().item()), 3),
        "mean_entropy": round(float(entropy.mean().item()), 3),
        "mean_max_probability": round(float(probs.max(dim=1).values.mean().item()), 3),
    }


def _standardize(train: torch.Tensor, test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True).clamp_min(1.0e-6)
    return (train - mean) / std, (test - mean) / std


def fit_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    task: ProbeTask,
    epochs: int = 120,
    lr: float = 5.0e-3,
    batch_size: int = 64,
    seed: int = 0,
) -> Dict[str, float | int | str]:
    generator = torch.Generator().manual_seed(seed)
    x = features.detach().float().cpu()
    if x.ndim > 2:
        x = x.flatten(start_dim=1)
    y = labels.detach().cpu()
    permutation = torch.randperm(x.shape[0], generator=generator)
    split = max(1, min(x.shape[0] - 1, int(x.shape[0] * 0.75)))
    train_idx = permutation[:split]
    test_idx = permutation[split:]
    x_train, x_test = _standardize(x[train_idx], x[test_idx])
    y_train, y_test = y[train_idx], y[test_idx]

    if task != "regression" and y_train.unique().numel() < 2:
        return {"status": "skipped", "reason": "single_class_train", "train_samples": int(y_train.numel())}

    if task == "regression":
        y_train_f = y_train.float()
        y_test_f = y_test.float()
        x_train_aug = torch.cat([x_train, torch.ones((x_train.shape[0], 1))], dim=1)
        x_test_aug = torch.cat([x_test, torch.ones((x_test.shape[0], 1))], dim=1)
        regularizer = torch.eye(x_train_aug.shape[1]) * 1.0e-2
        regularizer[-1, -1] = 0.0
        weights = torch.linalg.solve(
            x_train_aug.T @ x_train_aug + regularizer,
            x_train_aug.T @ y_train_f.unsqueeze(1),
        )
        predicted = (x_test_aug @ weights).squeeze(1)
        mse = torch.mean((predicted - y_test_f) ** 2)
        variance = torch.mean((y_test_f - y_test_f.mean()) ** 2).clamp_min(1.0e-8)
        r2 = 1.0 - mse / variance
        return {
            "status": "ok",
            "test_samples": int(y_test.numel()),
            "mse": round(float(mse.item()), 4),
            "r2": round(float(r2.item()), 3),
            "target_mean": round(float(y_test_f.mean().item()), 3),
            "target_std": round(float(y_test_f.std().item()) if y_test_f.numel() > 1 else 0.0, 3),
        }

    if task == "binary":
        model = nn.Linear(x_train.shape[1], 1)
        criterion: nn.Module = nn.BCEWithLogitsLoss()
        y_train_loss = y_train.float().unsqueeze(1)
    elif task == "multiclass":
        classes = int(y.max().item()) + 1
        model = nn.Linear(x_train.shape[1], classes)
        criterion = nn.CrossEntropyLoss()
        y_train_loss = y_train.long()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-3)
    for _epoch in range(epochs):
        batch_perm = torch.randperm(x_train.shape[0], generator=generator)
        for start in range(0, x_train.shape[0], batch_size):
            idx = batch_perm[start : start + batch_size]
            logits = model(x_train[idx])
            loss = criterion(logits, y_train_loss[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        pred = model(x_test)
        if task == "binary":
            predicted = (torch.sigmoid(pred.squeeze(1)) >= 0.5).long()
            accuracy = (predicted == y_test.long()).float().mean()
            baseline = max(float(y_test.float().mean().item()), 1.0 - float(y_test.float().mean().item()))
            return {
                "status": "ok",
                "test_samples": int(y_test.numel()),
                "positive_rate": round(float(y_test.float().mean().item()), 3),
                "accuracy": round(float(accuracy.item()), 3),
                "majority_baseline": round(baseline, 3),
            }
        if task == "multiclass":
            predicted = pred.argmax(dim=1)
            accuracy = (predicted == y_test.long()).float().mean()
            counts = Counter(int(value) for value in y_test.tolist())
            baseline = max(counts.values()) / y_test.numel()
            return {
                "status": "ok",
                "test_samples": int(y_test.numel()),
                "accuracy": round(float(accuracy.item()), 3),
                "majority_baseline": round(float(baseline), 3),
            }
        raise AssertionError(f"Unhandled probe task: {task}")


def build_analysis(
    model: PolicyValueNet,
    input_view: InputView,
    engine: RustEngine,
    samples: int,
    max_plies: int,
    seed: int,
    device: torch.device,
    probe_epochs: int,
    probe_lr: float,
    probe_batch_size: int,
) -> Dict[str, Any]:
    positions = collect_states(engine, samples=samples, max_plies=max_plies, seed=seed)
    boards: List[np.ndarray] = []
    sides: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    labels: Dict[str, List[float | int]] = {}
    for state, actions, item_labels in positions:
        board, side, mask = encode_state(state, engine, actions, input_view=input_view)
        boards.append(board)
        sides.append(side)
        masks.append(mask)
        for key, value in item_labels.items():
            labels.setdefault(key, []).append(value)

    board_tensor = torch.tensor(np.stack(boards), dtype=torch.float32, device=device)
    side_tensor = torch.tensor(np.stack(sides), dtype=torch.float32, device=device)
    mask_tensor = torch.tensor(np.stack(masks), dtype=torch.float32, device=device)
    activations = capture_activations(model, board_tensor, side_tensor)
    ranks = {name: effective_rank(value) for name, value in activations.items()}
    policy = legal_policy_summary(activations["move_logits"].cpu(), mask_tensor.cpu())

    probe_specs: Dict[str, ProbeTask] = {
        "current_is_red": "binary",
        "current_king_unprotected": "binary",
        "opponent_king_unprotected": "binary",
        "red_king_unprotected": "binary",
        "blue_king_unprotected": "binary",
        "has_legal_moves": "binary",
        "low_legal_count": "binary",
        "material_red_positive": "binary",
        "low_material": "binary",
        "phase": "multiclass",
        "legal_move_count": "regression",
        "material_balance_red": "regression",
        "total_pieces": "regression",
    }
    probe_layers = [
        layer
        for layer in ("board_flat", "side", "tokens_pre_transformer", "transformer_layer_1", "trunk_hidden")
        if layer in activations
    ]
    probes: Dict[str, Dict[str, Dict[str, float | int | str]]] = {}
    for layer in probe_layers:
        probes[layer] = {}
        for label_name, task in probe_specs.items():
            label_dtype = torch.long if task in ("binary", "multiclass") else torch.float32
            label_tensor = torch.tensor(labels[label_name], dtype=label_dtype)
            probes[layer][label_name] = fit_linear_probe(
                activations[layer].cpu(),
                label_tensor,
                task,
                epochs=probe_epochs,
                lr=probe_lr,
                batch_size=probe_batch_size,
                seed=seed + len(layer) + len(label_name),
            )

    return {
        "samples": len(positions),
        "max_plies": max_plies,
        "input_view": input_view,
        "label_distribution": {
            key: dict(Counter(int(value) for value in values)) if key != "material_balance_red" else {
                "mean": round(float(np.mean(values)), 3),
                "std": round(float(np.std(values)), 3),
            }
            for key, values in labels.items()
            if key in ("current_king_unprotected", "opponent_king_unprotected", "low_legal_count", "phase", "material_balance_red")
        },
        "activation_rank": ranks,
        "legal_policy": policy,
        "linear_probes": probes,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, input_view = load_model(
        args.checkpoint,
        device=device,
        hidden_size=args.hidden_size,
        model_arch=args.model_arch,
        input_view=args.input_view,
    )
    engine = RustEngine()
    try:
        analysis = build_analysis(
            model=model,
            input_view=input_view,
            engine=engine,
            samples=args.samples,
            max_plies=args.max_plies,
            seed=args.seed,
            device=device,
            probe_epochs=args.probe_epochs,
            probe_lr=args.probe_lr,
            probe_batch_size=args.probe_batch_size,
        )
    finally:
        engine.close()

    output = json.dumps(analysis, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
