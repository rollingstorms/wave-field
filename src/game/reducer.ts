import { cloneDefinitions, validateDefinition, validateDefinitions } from "../field/componentDefinitions";
import { playHeuristicTurn } from "./ai";
import { createInitialState, fromSnapshot } from "./initialState";
import { applyMove, applyTuning, beginTurn } from "./rules";
import type { BasisDefinition, Coefficient, GameState, PieceType, Position } from "./types";

export type GameAction =
  | { type: "select"; pieceId: string | null }
  | { type: "move"; pieceId: string; destination: Position }
  | { type: "tune"; pieceType: PieceType; componentIndex: number; value: Coefficient }
  | { type: "ai-turn" }
  | { type: "undo" }
  | { type: "restart"; keepDefinitions?: boolean }
  | { type: "update-definition"; pieceType: PieceType; componentIndex: number; definition: BasisDefinition }
  | { type: "reset-definitions" }
  | { type: "import-definitions"; payload: unknown };

export function gameReducer(state: GameState, action: GameAction): GameState {
  switch (action.type) {
    case "select":
      return { ...state, selectedPieceId: action.pieceId };
    case "move": {
      const result = applyMove(action.pieceId, action.destination, state);
      return result.ok ? result.state : { ...state, message: result.reason ?? state.message };
    }
    case "tune": {
      const result = applyTuning(state.currentPlayer, action.pieceType, action.componentIndex, action.value, state);
      return result.ok ? result.state : { ...state, message: result.reason ?? state.message };
    }
    case "ai-turn":
      return playHeuristicTurn(state);
    case "undo": {
      const previous = state.history.at(-1);
      if (!previous) return { ...state, message: "Nothing to undo" };
      return fromSnapshot(previous, state.history.slice(0, -1));
    }
    case "restart": {
      const initial = beginTurn(createInitialState());
      return action.keepDefinitions ? { ...initial, definitions: state.definitions } : initial;
    }
    case "update-definition": {
      if (!validateDefinition(action.definition)) {
        return { ...state, message: "Wave definitions must use base-2 decay and origin scale 1" };
      }
      const definitions = structuredClone(state.definitions);
      definitions[action.pieceType][action.componentIndex] = action.definition;
      return { ...state, definitions };
    }
    case "reset-definitions":
      return { ...state, definitions: cloneDefinitions() };
    case "import-definitions":
      return validateDefinitions(action.payload) ? { ...state, definitions: structuredClone(action.payload) } : { ...state, message: "Imported definitions are not valid" };
  }
}
