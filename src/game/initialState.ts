import { cloneDefinitions, DEFAULT_COMPONENTS } from "../field/componentDefinitions";
import { DEFAULT_HOME_ENERGY, DEFAULT_WAVE_SCALES } from "./constants";
import type { ComponentDefinitions, GameSnapshot, GameState, HomeEnergy, Piece, Player, PlayerComponents, PieceType, WaveScales } from "./types";

function piece(owner: Player, type: PieceType, x: number, y: number, n: number): Piece {
  return { id: `${owner}-${type}-${n}`, owner, type, position: { x, y }, unstable: false };
}

export function createInitialPieces(): Piece[] {
  return [
    piece("blue", "rook", 2, 0, 1),
    piece("blue", "king", 3, 0, 1),
    piece("blue", "rook", 4, 0, 2),
    piece("blue", "pawn", 2, 1, 1),
    piece("blue", "spy", 3, 1, 1),
    piece("blue", "pawn", 4, 1, 2),
    piece("red", "pawn", 2, 5, 1),
    piece("red", "spy", 3, 5, 1),
    piece("red", "pawn", 4, 5, 2),
    piece("red", "rook", 2, 6, 1),
    piece("red", "king", 3, 6, 1),
    piece("red", "rook", 4, 6, 2),
  ];
}

export function createInitialState(
  defaultComponents: PlayerComponents = DEFAULT_COMPONENTS,
  definitions: ComponentDefinitions = cloneDefinitions(),
  waveScales: WaveScales = structuredClone(DEFAULT_WAVE_SCALES),
  homeEnergy: HomeEnergy = structuredClone(DEFAULT_HOME_ENERGY),
): GameState {
  return {
    pieces: createInitialPieces(),
    currentPlayer: "blue",
    components: {
      blue: structuredClone(defaultComponents),
      red: structuredClone(defaultComponents),
    },
    defaultComponents: structuredClone(defaultComponents),
    definitions: structuredClone(definitions),
    waveScales: structuredClone(waveScales),
    homeEnergy: structuredClone(homeEnergy),
    selectedPieceId: null,
    status: "playing",
    history: [],
    turnNumber: 1,
    message: "Blue to move",
  };
}

export function snapshot(state: GameState): GameSnapshot {
  return {
    pieces: structuredClone(state.pieces),
    currentPlayer: state.currentPlayer,
    components: structuredClone(state.components),
    status: state.status,
    selectedPieceId: state.selectedPieceId,
    turnNumber: state.turnNumber,
    definitions: structuredClone(state.definitions),
    waveScales: structuredClone(state.waveScales),
    homeEnergy: structuredClone(state.homeEnergy),
  };
}

export function fromSnapshot(
  snap: GameSnapshot,
  history: GameSnapshot[] = [],
  defaultComponents: PlayerComponents = DEFAULT_COMPONENTS,
): GameState {
  return {
    ...structuredClone(snap),
    waveScales: structuredClone(snap.waveScales ?? DEFAULT_WAVE_SCALES),
    homeEnergy: structuredClone(snap.homeEnergy ?? DEFAULT_HOME_ENERGY),
    defaultComponents: structuredClone(defaultComponents),
    history,
    message: `${snap.currentPlayer === "blue" ? "Blue" : "Red"} to move`,
  };
}
