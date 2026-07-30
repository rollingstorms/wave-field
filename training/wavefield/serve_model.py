from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

import torch

from .engine import RustEngine
from .eval import load_model
from .selfplay import select_model_action
from .train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local-only Wave Field neural move API.")
    parser.add_argument("--checkpoint", type=Path, default=Path("training/runs/residual-rich-20m/checkpoint.pt"))
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--temperature", type=float, default=0.0)
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

    def move(self, state: Dict[str, Any], temperature: float | None = None) -> Dict[str, Any]:
        actions = self.engine.legal_actions(state)
        if not actions:
            raise ValueError("No legal neural moves are available.")
        action, _sample = select_model_action(
            self.model,
            state,
            self.engine,
            actions,
            temperature=self.args.temperature if temperature is None else temperature,
            device=self.device,
            record_sample=False,
            input_view=self.input_view,
        )
        return {
            "ok": True,
            "action": action,
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
