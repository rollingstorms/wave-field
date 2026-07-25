import type { GameState } from "./types";
import { fromSnapshot, snapshot } from "./initialState";

export function pushHistory(state: GameState): GameState {
  return { ...state, history: [...state.history, snapshot(state)] };
}

export function popHistory(state: GameState): GameState {
  const previous = state.history.at(-1);
  return previous ? fromSnapshot(previous, state.history.slice(0, -1)) : state;
}
