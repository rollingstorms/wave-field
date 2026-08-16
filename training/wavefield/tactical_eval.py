from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch

from .encoding import SIDE_SIZE, TUNING_ACTION_SIZE, board_channels_for_view
from .eval import load_model, numeric_distribution
from .engine import RustEngine
from .model import PolicyValueNet, masked_policy_logits
from .scenarios import DEFAULT_SCENARIOS, build_scenario_states, scenario_names
from .selfplay import Sample, heuristic_bootstrap_records
from .train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank tactical heuristic targets with a trained model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument(
        "--model-arch",
        choices=("conv", "residual", "transformer", "sequence_transformer", "encoder_sequence"),
        default=None,
    )
    parser.add_argument("--input-view", choices=("base", "piece_identity"), default=None)
    parser.add_argument("--history-plies", type=int, default=None)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--max-plies", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _rank(masked_logits: torch.Tensor, target_index: int) -> int:
    target_value = masked_logits[target_index]
    return int((masked_logits > target_value).sum().item()) + 1


def _reciprocal_rank(rank: int) -> float:
    return 1.0 / float(rank) if rank > 0 else 0.0


def _scenario(sample: Sample) -> str:
    return str(sample.metadata.get("scenario", "unknown"))


def _kind_name(action_kind: int) -> str:
    return "tune" if action_kind == 1 else "move"


def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def collect_tactical_samples(
    engine: RustEngine,
    scenarios: List[str],
    games: int,
    seed: int,
    input_view: str,
    max_plies: int = 1,
    history_plies: int = 1,
) -> List[Sample]:
    states = build_scenario_states(engine, scenarios, games=games, seed=seed)
    records = heuristic_bootstrap_records(
        engine,
        games=len(states),
        max_plies=max_plies,
        seed=seed + 1_000,
        input_view=input_view,
        initial_states=states,
        collect_metrics=False,
        history_plies=history_plies,
    )
    samples: List[Sample] = []
    for record in records:
        samples.extend(record.samples)
    return samples


def evaluate_tactical_samples(
    model: PolicyValueNet,
    samples: List[Sample],
    device: torch.device,
    batch_size: int = 128,
) -> Dict[str, Any]:
    model.eval()
    rows: List[Dict[str, Any]] = []

    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        boards = torch.tensor(np.stack([sample.board for sample in chunk]), dtype=torch.float32, device=device)
        sides = torch.tensor(np.stack([sample.side for sample in chunk]), dtype=torch.float32, device=device)
        move_masks = torch.tensor(np.stack([sample.legal_mask for sample in chunk]), dtype=torch.float32, device=device)
        tuning_masks = torch.tensor(
            np.stack([
                sample.legal_tuning_mask
                if sample.legal_tuning_mask is not None
                else np.zeros((TUNING_ACTION_SIZE,), dtype=np.float32)
                for sample in chunk
            ]),
            dtype=torch.float32,
            device=device,
        )
        history_boards = None
        history_sides = None
        if all(sample.history_board is not None and sample.history_side is not None for sample in chunk):
            history_boards = torch.tensor(
                np.stack([sample.history_board for sample in chunk if sample.history_board is not None]),
                dtype=torch.float32,
                device=device,
            )
            history_sides = torch.tensor(
                np.stack([sample.history_side for sample in chunk if sample.history_side is not None]),
                dtype=torch.float32,
                device=device,
            )

        with torch.no_grad():
            kind_logits, move_logits, tuning_logits = model.full_policy(
                boards,
                sides,
                history_board=history_boards,
                history_side=history_sides,
            )
            _legacy_logits, values = model(
                boards,
                sides,
                history_board=history_boards,
                history_side=history_sides,
            )
            legal_kind_mask = torch.stack(
                [
                    (move_masks.sum(dim=1) > 0).float(),
                    (tuning_masks.sum(dim=1) > 0).float(),
                ],
                dim=1,
            )
            kind_masked = masked_policy_logits(kind_logits, legal_kind_mask)
            kind_probs = torch.softmax(kind_masked, dim=1)
            move_masked = masked_policy_logits(move_logits, move_masks)
            tuning_masked = masked_policy_logits(tuning_logits, tuning_masks)

        for offset, sample in enumerate(chunk):
            target_kind = int(sample.action_kind)
            target_index = int(sample.tuning_action_index if target_kind == 1 else sample.action_index)
            masked = tuning_masked[offset] if target_kind == 1 else move_masked[offset]
            legal_count = int((tuning_masks[offset] if target_kind == 1 else move_masks[offset]).sum().item())
            if target_index < 0 or legal_count <= 0:
                continue
            rank = _rank(masked, target_index)
            rows.append(
                {
                    "scenario": _scenario(sample),
                    "kind": _kind_name(target_kind),
                    "target_kind": target_kind,
                    "kind_top1": int(kind_masked[offset].argmax().item()) == target_kind,
                    "kind_prob": float(kind_probs[offset, target_kind].item()),
                    "rank": rank,
                    "legal_count": legal_count,
                    "top1": rank == 1,
                    "top3": rank <= 3,
                    "top5": rank <= 5,
                    "mrr": _reciprocal_rank(rank),
                    "value": float(values[offset].item()),
                }
            )

    return summarize_tactical_rows(rows)


def summarize_tactical_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_key: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key["all"].append(row)
        by_key[f"scenario:{row['scenario']}"].append(row)
        by_key[f"kind:{row['kind']}"].append(row)
        by_key[f"scenario:{row['scenario']}:kind:{row['kind']}"].append(row)

    def summarize(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        ranks = [int(row["rank"]) for row in group]
        kinds = Counter(str(row["kind"]) for row in group)
        scenarios = Counter(str(row["scenario"]) for row in group)
        return {
            "samples": len(group),
            "kinds": dict(kinds),
            "scenarios": dict(scenarios),
            "kind_top1": _safe_mean(float(row["kind_top1"]) for row in group),
            "mean_kind_prob": _safe_mean(float(row["kind_prob"]) for row in group),
            "target_top1": _safe_mean(float(row["top1"]) for row in group),
            "target_top3": _safe_mean(float(row["top3"]) for row in group),
            "target_top5": _safe_mean(float(row["top5"]) for row in group),
            "target_mrr": _safe_mean(float(row["mrr"]) for row in group),
            "mean_rank": _safe_mean(float(row["rank"]) for row in group),
            "rank_distribution": numeric_distribution(ranks),
            "mean_legal_count": _safe_mean(float(row["legal_count"]) for row in group),
            "mean_value": _safe_mean(float(row["value"]) for row in group),
        }

    return {
        "samples": len(rows),
        "groups": {
            key: summarize(group)
            for key, group in sorted(by_key.items())
        },
    }


def run_tactical_eval(
    engine: RustEngine,
    model: PolicyValueNet,
    device: torch.device,
    input_view: str,
    scenarios: List[str],
    games: int,
    seed: int,
    max_plies: int = 1,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    samples = collect_tactical_samples(
        engine,
        scenarios=scenarios,
        games=games,
        seed=seed,
        input_view=input_view,
        max_plies=max_plies,
        history_plies=getattr(model, "history_plies", 1),
    )
    summary = evaluate_tactical_samples(model, samples, device)
    summary["seconds"] = round(time.perf_counter() - started_at, 3)
    summary["games"] = games
    summary["max_plies"] = max_plies
    summary["scenario_names"] = scenarios
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    all_group = summary["groups"].get("all", {})
    print(
        f"samples={summary['samples']} games={summary['games']} seconds={summary['seconds']} "
        f"kind_top1={all_group.get('kind_top1', 0.0):.3f} "
        f"target_top1={all_group.get('target_top1', 0.0):.3f} "
        f"target_top3={all_group.get('target_top3', 0.0):.3f} "
        f"mrr={all_group.get('target_mrr', 0.0):.3f} "
        f"mean_rank={all_group.get('mean_rank', 0.0):.2f}"
    )
    for key, group in summary["groups"].items():
        if key == "all" or ":kind:" in key:
            continue
        print(
            f"{key} samples={group['samples']} kind_top1={group['kind_top1']:.3f} "
            f"target_top1={group['target_top1']:.3f} target_top3={group['target_top3']:.3f} "
            f"mrr={group['target_mrr']:.3f} mean_rank={group['mean_rank']:.2f}"
        )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    engine = RustEngine()
    model, input_view = load_model(
        args.checkpoint,
        args.hidden_size,
        device,
        model_arch=args.model_arch,
        input_view=args.input_view,
        history_plies=args.history_plies,
    )
    summary = run_tactical_eval(
        engine,
        model,
        device,
        input_view=input_view,
        scenarios=scenario_names(args.scenarios),
        games=args.games,
        seed=args.seed,
        max_plies=args.max_plies,
    )
    engine.close()
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print_summary(summary)


if __name__ == "__main__":
    main()
