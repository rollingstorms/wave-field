from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

import torch

from .engine import RustEngine, TuningAction
from .encoding import (
    decode_tuning_action,
    encode_state,
    legal_tuning_actions,
    legal_tuning_mask,
    tuning_action_index,
)
from .eval import load_model
from .model import masked_policy_logits
from .selfplay import select_model_action
from .train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local-only Wave Field neural move API.")
    parser.add_argument("--checkpoint", type=Path, default=Path("training/runs/residual-rich-20m/checkpoint.pt"))
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tuning-temperature", type=float, default=None)
    parser.add_argument("--max-tuning-actions", type=int, default=3)
    parser.add_argument("--min-tuning-actions", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


class ModelMoveServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = resolve_device(args.device)
        self.engine = RustEngine()
        self.model, self.input_view = load_model(
            args.checkpoint,
            args.hidden_size,
            self.device,
        )

    def _select_tuning_action(self, state: Dict[str, Any], temperature: float) -> TuningAction | None:
        actions = legal_tuning_actions(state)
        if not actions:
            return None
        moves = self.engine.legal_actions(state)
        board, side, _move_mask = encode_state(state, self.engine, moves, input_view=self.input_view)
        board_tensor = torch.tensor(board, dtype=torch.float32, device=self.device).unsqueeze(0)
        side_tensor = torch.tensor(side, dtype=torch.float32, device=self.device).unsqueeze(0)
        tune_mask = torch.tensor(legal_tuning_mask(actions), dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            _kind_logits, _move_logits, tune_logits = self.model.full_policy(board_tensor, side_tensor)
            masked = masked_policy_logits(tune_logits, tune_mask).squeeze(0)
            if temperature <= 0:
                selected_index = int(masked.argmax().item())
            else:
                probs = torch.softmax(masked / temperature, dim=0)
                selected_index = int(torch.multinomial(probs, 1).item())

        legal_indexes = {tuning_action_index(action) for action in actions}
        if selected_index not in legal_indexes:
            selected_index = tuning_action_index(actions[0])
        return decode_tuning_action(selected_index)

    def _should_tune(self, state: Dict[str, Any], tune_count: int, temperature: float) -> bool:
        if tune_count < self.args.min_tuning_actions:
            return True
        if tune_count >= self.args.max_tuning_actions:
            return False
        tune_actions = legal_tuning_actions(state)
        move_actions = self.engine.legal_actions(state)
        if not tune_actions:
            return False
        if not move_actions:
            return True

        board, side, move_mask = encode_state(state, self.engine, move_actions, input_view=self.input_view)
        board_tensor = torch.tensor(board, dtype=torch.float32, device=self.device).unsqueeze(0)
        side_tensor = torch.tensor(side, dtype=torch.float32, device=self.device).unsqueeze(0)
        move_mask_tensor = torch.tensor(move_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        tune_mask_tensor = torch.tensor(legal_tuning_mask(tune_actions), dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            kind_logits, _move_logits, _tune_logits = self.model.full_policy(board_tensor, side_tensor)
            has_move = move_mask_tensor.sum(dim=1) > 0
            has_tune = tune_mask_tensor.sum(dim=1) > 0
            kind_mask = torch.stack([has_move, has_tune], dim=1).to(dtype=torch.float32)
            masked = masked_policy_logits(kind_logits, kind_mask).squeeze(0)
            if temperature <= 0:
                return int(masked.argmax().item()) == 1
            probs = torch.softmax(masked / temperature, dim=0)
            return int(torch.multinomial(probs, 1).item()) == 1

    def move(self, state: Dict[str, Any], temperature: float | None = None) -> Dict[str, Any]:
        action_temperature = self.args.temperature if temperature is None else temperature
        tuning_temperature = self.args.tuning_temperature
        if tuning_temperature is None:
            tuning_temperature = action_temperature
        tuned_state = state
        tuning_actions: List[TuningAction] = []
        while self._should_tune(tuned_state, len(tuning_actions), tuning_temperature):
            action = self._select_tuning_action(tuned_state, tuning_temperature)
            if action is None:
                break
            next_tuned_state = self.engine.apply_tuning(tuned_state, action)
            if not self.engine.legal_actions(next_tuned_state):
                break
            tuned_state = next_tuned_state
            tuning_actions.append(action)

        actions = self.engine.legal_actions(tuned_state)
        if not actions:
            terminal_state = self.engine.begin_turn(tuned_state, analyze_checkmate=True)
            if terminal_state.get("status") != "playing":
                return {
                    "ok": True,
                    "terminalState": terminal_state,
                    "actions": tuning_actions,
                    "tuningActions": len(tuning_actions),
                    "legalActions": 0,
                    "inputView": self.input_view,
                    "checkpoint": str(self.args.checkpoint),
                }
            raise ValueError("No legal neural moves are available.")
        action, _sample = select_model_action(
            self.model,
            tuned_state,
            self.engine,
            actions,
            temperature=action_temperature,
            device=self.device,
            record_sample=False,
            input_view=self.input_view,
        )
        move_action = {"type": "move", **action}
        return {
            "ok": True,
            "action": action,
            "actions": [*tuning_actions, move_action],
            "tuningActions": len(tuning_actions),
            "legalActions": len(actions),
            "inputView": self.input_view,
            "checkpoint": str(self.args.checkpoint),
        }

    def close(self) -> None:
        self.engine.close()


def make_handler(server_state: ModelMoveServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "content-type")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:
            self._send_json(200, {"ok": True})

        def do_GET(self) -> None:
            if self.path != "/health":
                self._send_json(404, {"ok": False, "error": "Not found"})
                return
            self._send_json(
                200,
                {
                    "ok": True,
                    "checkpoint": str(server_state.args.checkpoint),
                    "inputView": server_state.input_view,
                    "device": str(server_state.device),
                },
            )

        def do_POST(self) -> None:
            if self.path != "/move":
                self._send_json(404, {"ok": False, "error": "Not found"})
                return
            try:
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                state = payload["state"]
                temperature = payload.get("temperature")
                response = server_state.move(
                    state,
                    temperature=float(temperature) if temperature is not None else None,
                )
                self._send_json(200, response)
            except Exception as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main() -> None:
    args = parse_args()
    model_server = ModelMoveServer(args)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(model_server))
    print(
        json.dumps(
            {
                "event": "serving",
                "url": f"http://{args.host}:{args.port}",
                "checkpoint": str(args.checkpoint),
                "inputView": model_server.input_view,
                "device": str(model_server.device),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        httpd.serve_forever()
    finally:
        model_server.close()


if __name__ == "__main__":
    main()
