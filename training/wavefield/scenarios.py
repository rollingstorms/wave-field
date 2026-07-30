from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Sequence

from .engine import RustEngine, load_initial_state


ScenarioName = str

DEFAULT_SCENARIOS = ("opening", "midgame", "low_material", "rescue")


def scenario_names(raw: str) -> List[ScenarioName]:
    if not raw:
        return []
    return [name.strip() for name in raw.split(",") if name.strip()]


def build_scenario_states(
    engine: RustEngine,
    names: Sequence[ScenarioName],
    games: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if games <= 0 or not names:
        return []

    rng = random.Random(seed)
    states: List[Dict[str, Any]] = []
    for index in range(games):
        name = names[index % len(names)]
        states.append(build_scenario_state(engine, name, rng))
    return states


def build_scenario_state(
    engine: RustEngine,
    name: ScenarioName,
    rng: random.Random,
) -> Dict[str, Any]:
    if name == "opening":
        state = load_initial_state()
    elif name == "midgame":
        state = _random_advance(engine, load_initial_state(), plies=24, rng=rng)
    elif name == "low_material":
        state = _low_material_state()
    elif name == "rescue":
        state = _rescue_state()
    else:
        raise ValueError(f"Unknown scenario '{name}'. Known scenarios: {', '.join(DEFAULT_SCENARIOS)}")

    state.setdefault("metadata", {})
    state["metadata"] = {**state["metadata"], "scenario": name}
    return state


def _random_advance(
    engine: RustEngine,
    state: Dict[str, Any],
    plies: int,
    rng: random.Random,
) -> Dict[str, Any]:
    current = copy.deepcopy(state)
    for _ply in range(plies):
        if current["status"] != "playing":
            break
        actions = engine.legal_actions(current)
        if not actions:
            break
        current = engine.apply_action(current, rng.choice(actions), analyze_checkmate=False)
    return current


def _low_material_state() -> Dict[str, Any]:
    state = copy.deepcopy(load_initial_state())
    keep = {
        "blue-king-1",
        "blue-rook-1",
        "blue-spy-1",
        "red-king-1",
        "red-rook-1",
        "red-spy-1",
    }
    state["pieces"] = [piece for piece in state["pieces"] if piece["id"] in keep]
    state["currentPlayer"] = "blue"
    return state


def _rescue_state() -> Dict[str, Any]:
    state = copy.deepcopy(load_initial_state())
    state["currentPlayer"] = "red"
    for piece in state["pieces"]:
        if piece["id"] == "red-spy-1":
            piece["unstable"] = True
        elif piece["owner"] == "red":
            piece["unstable"] = False
    return state
