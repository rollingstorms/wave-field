from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TypedDict


REPO_ROOT = Path(__file__).resolve().parents[2]
INITIAL_STATE_PATH = REPO_ROOT / "engine/tests/initial-state.json"


class Position(TypedDict):
    x: int
    y: int


class Action(TypedDict):
    pieceId: str
    destination: Position


def load_initial_state(path: Path = INITIAL_STATE_PATH) -> Dict[str, Any]:
    return json.loads(path.read_text())


@dataclass
class RustEngine:
    binary: Path = REPO_ROOT / "engine/target/release/wave-field-engine"

    def build_release(self) -> None:
        subprocess.run(
            ["cargo", "build", "--manifest-path", "engine/Cargo.toml", "--release"],
            cwd=REPO_ROOT,
            check=True,
        )

    def request(self, method: str, state: Dict[str, Any], **params: Any) -> Any:
        payload = {"method": method, "state": state, **params}
        try:
            result = subprocess.run(
                [str(self.binary)],
                cwd=REPO_ROOT,
                input=json.dumps(payload) + "\n",
                text=True,
                capture_output=True,
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Rust engine binary is missing. Run: "
                "cargo build --manifest-path engine/Cargo.toml --release"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Rust engine failed:\n{exc.stderr}") from exc
        return json.loads(result.stdout)

    def evaluate_field(self, state: Dict[str, Any]) -> List[List[float]]:
        return self.request("evaluateField", state)

    def legal_moves(self, state: Dict[str, Any], piece_id: str) -> List[Position]:
        return self.request("legalMoves", state, pieceId=piece_id)

    def playable_moves(self, state: Dict[str, Any], piece_id: str) -> List[Position]:
        return self.request("playableMoves", state, pieceId=piece_id)

    def legal_actions(self, state: Dict[str, Any], playable: bool = True) -> List[Action]:
        actions: List[Action] = []
        current_player = state["currentPlayer"]
        for piece in state["pieces"]:
            if piece["owner"] != current_player:
                continue
            moves: Iterable[Position] = (
                self.playable_moves(state, piece["id"])
                if playable
                else self.legal_moves(state, piece["id"])
            )
            for destination in moves:
                actions.append({"pieceId": piece["id"], "destination": destination})
        return actions

    def apply_action(
        self,
        state: Dict[str, Any],
        action: Action,
        analyze_checkmate: bool = False,
    ) -> Dict[str, Any]:
        result = self.request(
            "applyMove",
            state,
            pieceId=action["pieceId"],
            destination=action["destination"],
            analyzeCheckmate=analyze_checkmate,
        )
        if not result["ok"]:
            raise ValueError(result.get("reason") or "Rust engine rejected action")
        return result["state"]

    def play_heuristic_turn(
        self,
        state: Dict[str, Any],
        player: Optional[str] = None,
        seed: int = 0,
        variety: float = 0.55,
        time_budget_ms: int = 10,
    ) -> Dict[str, Any]:
        return self.request(
            "playHeuristicTurn",
            state,
            player=player or state["currentPlayer"],
            seed=seed,
            variety=variety,
            timeBudgetMs=time_budget_ms,
        )
