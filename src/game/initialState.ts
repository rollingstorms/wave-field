import { cloneDefinitions, DEFAULT_COMPONENTS } from "../field/componentDefinitions";
import { evaluateInstantField } from "../field/evaluateField";
import { BOARD_CENTER, BOARD_SIZE, DEFAULT_HOME_ENERGY, DEFAULT_WAVE_SCALES, SPARSE_HOME_ENERGY } from "./constants";
import { activationOrdersForPlayers } from "./tuning";
import { createAmpSquares } from "./variants";
import type { ComponentDefinitions, GameSnapshot, GameState, GameVariant, HomeEnergy, Piece, Player, PlayerComponents, PieceType, WaveScales } from "./types";

const routePath = globalThis.location?.pathname.replace(/\/$/, "") ?? "";
const ampRouteEnabled = routePath.endsWith("/amp");
const entropyRouteEnabled = routePath.endsWith("/entropy");
const continuousRouteEnabled = routePath.endsWith("/continuous");
const classicRoutes = ["classic", "amp", "entropy", "big", "wavefielder", "local-arena", "optim-test", "easy-test", "low-rescue-test"];
const routeVariant: GameVariant = continuousRouteEnabled ? "continuous"
  : classicRoutes.some((route) => routePath.endsWith(`/${route}`)) ? "classic" : "sparse";

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
  homeEnergy?: HomeEnergy,
  ampSquares = ampRouteEnabled ? createAmpSquares() : [],
  variant: GameVariant = routeVariant,
): GameState {
  const components = {
    blue: structuredClone(defaultComponents),
    red: structuredClone(defaultComponents),
  };
  const state: GameState = {
    variant,
    pieces: createInitialPieces(),
    currentPlayer: "blue",
    components,
    activationOrders: activationOrdersForPlayers(components),
    defaultComponents: structuredClone(defaultComponents),
    definitions: structuredClone(definitions),
    waveScales: structuredClone(waveScales),
    homeEnergy: structuredClone(homeEnergy ?? (variant === "sparse" ? SPARSE_HOME_ENERGY : DEFAULT_HOME_ENERGY)),
    ampSquares: structuredClone(ampSquares),
    selectedPieceId: null,
    status: "playing",
    history: [],
    turnNumber: 1,
    message: "Blue to move",
  };
  return entropyRouteEnabled ? { ...state, entropyField: evaluateInstantField(state) } : state;
}

export function createClassicInitialState(
  defaultComponents?: PlayerComponents,
  definitions?: ComponentDefinitions,
  waveScales?: WaveScales,
  homeEnergy?: HomeEnergy,
  ampSquares = ampRouteEnabled ? createAmpSquares() : [],
): GameState {
  return createInitialState(defaultComponents, definitions, waveScales, homeEnergy, ampSquares, "classic");
}

export function snapshot(state: GameState): GameSnapshot {
  const snap: GameSnapshot = {
    variant: state.variant,
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
    ampSquares: structuredClone(state.ampSquares),
  };
  if (state.entropyField) snap.entropyField = structuredClone(state.entropyField);
  return snap;
}

export function fromSnapshot(
  snap: GameSnapshot,
  history: GameSnapshot[] = [],
  defaultComponents: PlayerComponents = DEFAULT_COMPONENTS,
): GameState {
  return {
    ...structuredClone(snap),
    // Missing variants belong to saves created before discrete rules became the default.
    variant: snap.variant ?? "classic",
    activationOrders: structuredClone(snap.activationOrders ?? activationOrdersForPlayers(snap.components)),
    waveScales: structuredClone(snap.waveScales ?? DEFAULT_WAVE_SCALES),
    homeEnergy: structuredClone(snap.homeEnergy ?? DEFAULT_HOME_ENERGY),
    ampSquares: structuredClone(snap.ampSquares ?? []),
    entropyField: snap.entropyField ? structuredClone(snap.entropyField) : undefined,
    defaultComponents: structuredClone(defaultComponents),
    history,
    message: `${snap.currentPlayer === "blue" ? "Blue" : "Red"} to move`,
  };
}
