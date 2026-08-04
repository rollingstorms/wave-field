import { cloneDefinitions, DEFAULT_COMPONENTS } from "../field/componentDefinitions";
import { BOARD_CENTER, BOARD_SIZE, DEFAULT_HOME_ENERGY, DEFAULT_WAVE_SCALES } from "./constants";
import { activationOrdersForPlayers } from "./tuning";
import type { ComponentDefinitions, GameSnapshot, GameState, HomeEnergy, Piece, Player, PlayerComponents, PieceType, WaveScales } from "./types";

function piece(owner: Player, type: PieceType, x: number, y: number, n: number): Piece {
  return { id: `${owner}-${type}-${n}`, owner, type, position: { x, y }, unstable: false };
}

export function createInitialPieces(): Piece[] {
  const left = BOARD_CENTER - 1;
  const middle = BOARD_CENTER;
  const right = BOARD_CENTER + 1;
  const redBackRank = BOARD_SIZE - 1;
  const redPawnRank = BOARD_SIZE - 2;
  return [
    piece("blue", "rook", left, 0, 1),
    piece("blue", "king", middle, 0, 1),
    piece("blue", "rook", right, 0, 2),
    piece("blue", "pawn", left, 1, 1),
    piece("blue", "spy", middle, 1, 1),
    piece("blue", "pawn", right, 1, 2),
    piece("red", "pawn", left, redPawnRank, 1),
    piece("red", "spy", middle, redPawnRank, 1),
    piece("red", "pawn", right, redPawnRank, 2),
    piece("red", "rook", left, redBackRank, 1),
    piece("red", "king", middle, redBackRank, 1),
    piece("red", "rook", right, redBackRank, 2),
  ];
}

export function createInitialState(
  defaultComponents: PlayerComponents = DEFAULT_COMPONENTS,
  definitions: ComponentDefinitions = cloneDefinitions(),
  waveScales: WaveScales = structuredClone(DEFAULT_WAVE_SCALES),
  homeEnergy: HomeEnergy = structuredClone(DEFAULT_HOME_ENERGY),
): GameState {
  const components = {
    blue: structuredClone(defaultComponents),
    red: structuredClone(defaultComponents),
  };
  return {
    pieces: createInitialPieces(),
    currentPlayer: "blue",
    components,
    activationOrders: activationOrdersForPlayers(components),
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
    activationOrders: structuredClone(state.activationOrders),
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
    activationOrders: structuredClone(snap.activationOrders ?? activationOrdersForPlayers(snap.components)),
    waveScales: structuredClone(snap.waveScales ?? DEFAULT_WAVE_SCALES),
    homeEnergy: structuredClone(snap.homeEnergy ?? DEFAULT_HOME_ENERGY),
    defaultComponents: structuredClone(defaultComponents),
    history,
    message: `${snap.currentPlayer === "blue" ? "Blue" : "Red"} to move`,
  };
}
