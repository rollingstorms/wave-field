import { cloneDefinitions, DEFAULT_COMPONENTS, validateDefinition, validateDefinitions } from "../field/componentDefinitions";
import { DEFAULT_HOME_ENERGY, DEFAULT_WAVE_SCALES, TUNING_STRENGTH } from "./constants";
import { playHeuristicTurn } from "./ai";
import { createInitialState, fromSnapshot, snapshot } from "./initialState";
import { applyClosestPlayableHint, applyMove, applyTuning, beginTurn, randomizeTuning, resetTuning, resignInCheck } from "./rules";
import type { BasisDefinition, Coefficient, GameState, PieceType, Player, Position } from "./types";

export type GameAction =
  | { type: "select"; pieceId: string | null }
  | { type: "move"; pieceId: string; destination: Position }
  | { type: "tune"; pieceType: PieceType; componentIndex: number; value: Coefficient }
  | { type: "resign" }
  | { type: "hint" }
  | { type: "randomize-tuning" }
  | { type: "reset-tuning" }
  | { type: "ai-turn"; player?: Player; seed?: number; variety?: number }
  | { type: "undo" }
  | { type: "restart"; keepDefinitions?: boolean }
  | { type: "update-default-component"; pieceType: PieceType; componentIndex: number; value: Coefficient }
  | { type: "update-wave-scale"; pieceType: PieceType; scale: "friendly" | "hostile"; value: number }
  | { type: "reset-wave-scales" }
  | { type: "update-home-energy"; pieceType: PieceType; value: number }
  | { type: "reset-home-energy" }
  | { type: "reset-default-components" }
  | { type: "update-definition"; pieceType: PieceType; componentIndex: number; definition: BasisDefinition }
  | { type: "reset-definitions" }
  | { type: "import-definitions"; payload: unknown };

function setDefaultControlAtStrength(
  defaults: GameState["defaultComponents"],
  pieceType: PieceType,
  componentIndex: number,
  value: Coefficient,
): GameState["defaultComponents"] | null {
  if (value === 0) return null;
  const next = structuredClone(defaults);
  const coefficients = next[pieceType];
  if (coefficients[componentIndex] === value) return null;

  const activeIndices = coefficients.flatMap((coefficient, index) => coefficient === 0 ? [] : [index]);
  if (coefficients[componentIndex] === 0 && activeIndices.length >= TUNING_STRENGTH[pieceType]) {
    const evictedIndex = activeIndices.find((index) => index !== componentIndex);
    if (evictedIndex !== undefined) coefficients[evictedIndex] = 0;
  }
  coefficients[componentIndex] = value;
  return next;
}

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
    case "resign": {
      const result = resignInCheck(state);
      return result.ok ? result.state : { ...state, message: result.reason ?? state.message };
    }
    case "hint": {
      const result = applyClosestPlayableHint(state);
      return result.ok ? result.state : { ...state, message: result.reason ?? state.message };
    }
    case "randomize-tuning": {
      const result = randomizeTuning(state);
      return result.ok ? result.state : { ...state, message: result.reason ?? state.message };
    }
    case "reset-tuning": {
      const result = resetTuning(state);
      return result.ok ? result.state : { ...state, message: result.reason ?? state.message };
    }
    case "ai-turn":
      return playHeuristicTurn(state, action.player ?? "red", { seed: action.seed, variety: action.variety });
    case "undo": {
      const previous = state.history.at(-1);
      if (!previous) return { ...state, message: "Nothing to undo" };
      return fromSnapshot(previous, state.history.slice(0, -1), state.defaultComponents);
    }
    case "restart": {
      const definitions = action.keepDefinitions ? state.definitions : cloneDefinitions();
      return beginTurn(createInitialState(state.defaultComponents, definitions, state.waveScales, state.homeEnergy));
    }
    case "update-wave-scale": {
      if (!Number.isFinite(action.value) || action.value < 0 || action.value > 4) {
        return { ...state, message: "Wave scales must be between 0 and 4." };
      }
      const waveScales = structuredClone(state.waveScales);
      waveScales[action.pieceType][action.scale] = action.value;
      return {
        ...state,
        waveScales,
        history: [...state.history, snapshot(state)],
        message: `${action.pieceType} ${action.scale} scale set to ${action.value.toFixed(2)}`,
      };
    }
    case "reset-wave-scales":
      return { ...state, waveScales: structuredClone(DEFAULT_WAVE_SCALES), history: [...state.history, snapshot(state)], message: "Wave scales reset" };
    case "update-home-energy": {
      if (!Number.isFinite(action.value) || action.value < -4 || action.value > 4) {
        return { ...state, message: "Home energy must be between -4 and 4." };
      }
      const homeEnergy = structuredClone(state.homeEnergy);
      homeEnergy[action.pieceType] = action.value;
      return {
        ...state,
        homeEnergy,
        history: [...state.history, snapshot(state)],
        message: `${action.pieceType} home energy set to ${action.value.toFixed(2)}`,
      };
    }
    case "reset-home-energy":
      return { ...state, homeEnergy: structuredClone(DEFAULT_HOME_ENERGY), history: [...state.history, snapshot(state)], message: "Home energy reset" };
    case "update-default-component": {
      const defaultComponents = setDefaultControlAtStrength(state.defaultComponents, action.pieceType, action.componentIndex, action.value);
      if (!defaultComponents) return { ...state, message: "Default controls must stay at full strength." };
      return { ...state, defaultComponents, message: "Default controls updated · restart to apply" };
    }
    case "reset-default-components":
      return { ...state, defaultComponents: structuredClone(DEFAULT_COMPONENTS), message: "Default controls reset · restart to apply" };
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
