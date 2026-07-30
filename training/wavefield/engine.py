from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
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
    persistent: bool = True
    _process: Optional[subprocess.Popen[str]] = field(default=None, init=False, repr=False)

    def build_release(self) -> None:
        subprocess.run(
            ["cargo", "build", "--manifest-path", "engine/Cargo.toml", "--release"],
            cwd=REPO_ROOT,
            check=True,
        )

    def request(self, method: str, state: Optional[Dict[str, Any]] = None, **params: Any) -> Any:
        payload = {"method": method, **params}
        if state is not None:
            payload["state"] = state
        if self.persistent:
            return self._persistent_request(payload)
        return self._one_shot_request(payload)

    def _one_shot_request(self, payload: Dict[str, Any]) -> Any:
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

    def _persistent_request(self, payload: Dict[str, Any]) -> Any:
        process = self._ensure_process()
        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            stderr = process.stderr.read() if process.stderr else ""
            self.close()
            raise RuntimeError(f"Rust engine stopped without a response:\n{stderr}")
        return json.loads(line)

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        try:
            self._process = subprocess.Popen(
                [str(self.binary)],
                cwd=REPO_ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Rust engine binary is missing. Run: "
                "cargo build --manifest-path engine/Cargo.toml --release"
            ) from exc
        return self._process

    def close(self) -> None:
        if self._process is None:
            return
        process = self._process
        self._process = None
        if process.poll() is None:
            if process.stdin:
                process.stdin.close()
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()

    def evaluate_field(self, state: Dict[str, Any]) -> List[List[float]]:
        return self.request("evaluateField", state)

    def legal_moves(self, state: Dict[str, Any], piece_id: str) -> List[Position]:
        return self.request("legalMoves", state, pieceId=piece_id)

    def playable_moves(self, state: Dict[str, Any], piece_id: str) -> List[Position]:
        return self.request("playableMoves", state, pieceId=piece_id)

    def legal_actions(self, state: Dict[str, Any], playable: bool = True) -> List[Action]:
        if playable:
            return self.request("playableActions", state)
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

    def player_actions(self, state: Dict[str, Any], player: str, playable: bool = True) -> List[Action]:
        probe = {**state, "currentPlayer": player}
        return self.legal_actions(probe, playable=playable)

    def unstable_piece_ids(self, state: Dict[str, Any], player: str) -> List[str]:
        return self.request("unstablePieceIds", state, player=player)

    def king_unprotected(self, state: Dict[str, Any], player: str) -> bool:
        return bool(self.request("kingUnprotected", state, player=player))

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

    def simulate_random_lean_games(
        self,
        state: Dict[str, Any],
        games: int,
        max_plies: int,
        seed: int = 0,
    ) -> Any:
        return self.request("simulateRandomLeanGames", state, games=games, maxPlies=max_plies, seed=seed)

    def generate_random_training_batch(
        self,
        state: Dict[str, Any],
        games: int,
        max_plies: int,
        seed: int = 0,
        material_for_capped: bool = True,
    ) -> Any:
        return self.request(
            "generateRandomTrainingBatch",
            state,
            games=games,
            maxPlies=max_plies,
            seed=seed,
            materialForCapped=material_for_capped,
        )

    def profile_random_training_batch(
        self,
        state: Dict[str, Any],
        games: int,
        max_plies: int,
        seed: int = 0,
        material_for_capped: bool = True,
    ) -> Any:
        return self.request(
            "profileRandomTrainingBatch",
            state,
            games=games,
            maxPlies=max_plies,
            seed=seed,
            materialForCapped=material_for_capped,
        )

    def create_rollout_session(
        self,
        states: List[Dict[str, Any]],
        max_plies: int,
    ) -> int:
        result = self.request("createRolloutSession", states=states, maxPlies=max_plies)
        return int(result["sessionId"])

    def get_rollout_batch(self, session_id: int) -> Dict[str, Any]:
        return self.request("getRolloutBatch", sessionId=session_id)

    def apply_rollout_actions(self, session_id: int, actions: List[Dict[str, int]]) -> Dict[str, Any]:
        return self.request("applyRolloutActions", sessionId=session_id, actions=actions)

    def finish_rollout_session(self, session_id: int) -> Dict[str, Any]:
        return self.request("finishRolloutSession", sessionId=session_id)
