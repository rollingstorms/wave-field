from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .encoding import (
    TUNING_ACTION_SIZE,
    action_index,
    decode_tuning_action,
    encode_state,
    legal_tuning_actions,
    legal_tuning_mask,
)
from .eval import load_model
from .engine import RustEngine
from .model import masked_policy_logits
from .scenarios import DEFAULT_SCENARIOS, build_scenario_states, scenario_names
from .train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect top legal move/tune logits for model positions.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument(
        "--model-arch",
        choices=("conv", "residual", "transformer", "sequence_transformer", "encoder_sequence"),
        default=None,
    )
    parser.add_argument("--input-view", choices=("base", "piece_identity"), default=None)
    parser.add_argument("--history-plies", type=int, default=None)
    parser.add_argument("--positions", type=int, default=8)
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _piece_by_id(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {piece["id"]: piece for piece in state["pieces"]}


def _describe_move(state: Dict[str, Any], action: Dict[str, Any], logit: float, prob: float) -> Dict[str, Any]:
    piece = _piece_by_id(state).get(action["pieceId"], {})
    return {
        "pieceId": action["pieceId"],
        "pieceType": piece.get("type"),
        "owner": piece.get("owner"),
        "from": piece.get("position"),
        "to": action["destination"],
        "logit": logit,
        "prob": prob,
    }


def _describe_tune(state: Dict[str, Any], index: int, logit: float, prob: float) -> Dict[str, Any]:
    action = decode_tuning_action(index)
    current = state["components"][state["currentPlayer"]][action["pieceType"]][action["componentIndex"]]
    return {
        **action,
        "currentValue": current,
        "logit": logit,
        "prob": prob,
    }


def _top_indexes(masked_logits: torch.Tensor, mask: torch.Tensor, top_k: int) -> List[int]:
    legal_count = int(mask.sum().item())
    if legal_count <= 0:
        return []
    count = min(top_k, legal_count)
    return [int(index) for index in torch.topk(masked_logits, k=count).indices.tolist()]


def inspect_positions(
    engine: RustEngine,
    model: torch.nn.Module,
    device: torch.device,
    input_view: str,
    scenarios: List[str],
    positions: int,
    seed: int,
    top_k: int,
) -> Dict[str, Any]:
    states = build_scenario_states(engine, scenarios, games=positions, seed=seed)
    rows = []
    model.eval()

    for state in states:
        move_actions = engine.legal_actions(state)
        tune_actions = legal_tuning_actions(state)
        board, side, move_mask = encode_state(state, engine, move_actions, input_view=input_view)
        tune_mask = legal_tuning_mask(tune_actions) if tune_actions else np.zeros((TUNING_ACTION_SIZE,), dtype=np.float32)

        board_tensor = torch.tensor(board, dtype=torch.float32, device=device).unsqueeze(0)
        side_tensor = torch.tensor(side, dtype=torch.float32, device=device).unsqueeze(0)
        move_mask_tensor = torch.tensor(move_mask, dtype=torch.float32, device=device).unsqueeze(0)
        tune_mask_tensor = torch.tensor(tune_mask, dtype=torch.float32, device=device).unsqueeze(0)
        kind_mask = torch.tensor(
            [[1.0 if move_actions else 0.0, 1.0 if tune_actions else 0.0]],
            dtype=torch.float32,
            device=device,
        )

        with torch.no_grad():
            kind_logits, move_logits, tune_logits = model.full_policy(board_tensor, side_tensor)
            _legacy_logits, value = model(board_tensor, side_tensor)
            kind_masked = masked_policy_logits(kind_logits, kind_mask).squeeze(0)
            kind_probs = torch.softmax(kind_masked, dim=0)
            move_masked = masked_policy_logits(move_logits, move_mask_tensor).squeeze(0)
            move_probs = torch.softmax(move_masked, dim=0)
            tune_masked = masked_policy_logits(tune_logits, tune_mask_tensor).squeeze(0)
            tune_probs = torch.softmax(tune_masked, dim=0)

        legal_moves_by_index = {action_index(action): action for action in move_actions}
        rows.append(
            {
                "scenario": state.get("metadata", {}).get("scenario", "initial"),
                "currentPlayer": state["currentPlayer"],
                "turnNumber": state.get("turnNumber"),
                "value": float(value.squeeze(0).item()),
                "kind": {
                    "move": {
                        "logit": float(kind_masked[0].item()),
                        "prob": float(kind_probs[0].item()),
                    },
                    "tune": {
                        "logit": float(kind_masked[1].item()),
                        "prob": float(kind_probs[1].item()),
                    },
                },
                "topMoves": [
                    _describe_move(
                        state,
                        legal_moves_by_index[index],
                        float(move_masked[index].item()),
                        float(move_probs[index].item()),
                    )
                    for index in _top_indexes(move_masked, move_mask_tensor.squeeze(0), top_k)
                ],
                "topTunes": [
                    _describe_tune(
                        state,
                        index,
                        float(tune_masked[index].item()),
                        float(tune_probs[index].item()),
                    )
                    for index in _top_indexes(tune_masked, tune_mask_tensor.squeeze(0), top_k)
                ],
            }
        )

    return {"positions": len(rows), "topK": top_k, "rows": rows}


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
    result = inspect_positions(
        engine,
        model,
        device,
        input_view=input_view,
        scenarios=scenario_names(args.scenarios),
        positions=args.positions,
        seed=args.seed,
        top_k=args.top_k,
    )
    engine.close()
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
