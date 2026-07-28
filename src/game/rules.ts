import { evaluateField } from "../field/evaluateField";
import { BOARD_SIZE, TUNING_STRENGTH } from "./constants";
import type { Coefficient, GameState, MoveResult, PieceType, Player, PlayerComponents, Position } from "./types";
import { getLegalMoves, samePosition } from "./movement";
import { activationOrderForProfile, isTuningAtStrength } from "./tuning";
import { getUnstablePieces, isKingUnprotected, markInstability, removeUnrescuedPieces } from "./victory";
import { snapshot } from "./initialState";
import {
  rustApplyClosestPlayableHint,
  rustApplyMove,
  rustApplyTuning,
  rustBeginTurn,
  rustClosestPlayableConfiguration,
  rustPlayableMoves,
  rustRandomizeTuning,
  rustResetTuning,
  rustResignInCheck,
} from "./rustEngine";

export function opponent(player: Player): Player {
  return player === "red" ? "blue" : "red";
}

function winStatus(winner: Player) {
  return winner === "red" ? "red-won" as const : "blue-won" as const;
}

function playerName(player: Player): string {
  return player === "red" ? "Red" : "Blue";
}

function boardCoordinate(position: Position): string {
  return `${String.fromCharCode(65 + position.x)}${BOARD_SIZE - position.y}`;
}

const pieceTypes: PieceType[] = ["pawn", "rook", "spy", "king"];
const coefficientValues: Coefficient[] = [1, 0, -1];

export interface PlayableConfigurationHint {
  components: PlayerComponents;
  pieceId: string;
  pieceType: PieceType;
  destination: Position;
  changedComponents: number;
}

interface RuleOptions {
  analyzeCheckmate?: boolean;
}

function componentOptions(pieceType: PieceType, count: number): Coefficient[][] {
  const options: Coefficient[][] = [];
  function build(values: Coefficient[]) {
    if (values.length === count) {
      if (isTuningAtStrength(pieceType, values)) options.push(values);
      return;
    }
    for (const value of coefficientValues) build([...values, value]);
  }
  build([]);
  return options;
}

function movePiece(state: GameState, pieceId: string, destination: Position): GameState {
  return {
    ...state,
    pieces: state.pieces.map((piece) => piece.id === pieceId ? { ...piece, position: destination } : piece),
  };
}

function resolveOwnTurnConsequences(player: Player, previous: GameState, candidate: GameState): GameState {
  const previousField = evaluateField(previous);
  const rescueDeadlineIds = new Set(
    previous.pieces
      .filter((piece) => piece.owner === player && piece.type !== "king" && piece.unstable)
      .map((piece) => piece.id),
  );
  for (const piece of getUnstablePieces(player, previous, previousField).filter((piece) => piece.type !== "king")) {
    rescueDeadlineIds.add(piece.id);
  }
  const marked = markInstability(candidate, evaluateField(candidate));
  const deadlineResolved = removeUnrescuedPieces(player, marked, rescueDeadlineIds);
  const selfField = evaluateField(deadlineResolved);
  return markInstability(deadlineResolved, selfField);
}

function lostOwnPieces(player: Player, before: GameState, after: GameState): string[] {
  const remainingIds = new Set(after.pieces.map((piece) => piece.id));
  return before.pieces
    .filter((piece) => piece.owner === player && piece.type !== "king" && !remainingIds.has(piece.id))
    .map((piece) => piece.type);
}

function componentDistance(left: PlayerComponents, right: PlayerComponents): number {
  return pieceTypes.reduce((distance, pieceType) =>
    distance + left[pieceType].filter((value, index) => value !== right[pieceType][index]).length, 0);
}

function allComponentOptions(state: GameState, player: Player): PlayerComponents[] {
  const current = state.components[player];
  const profiles = Object.fromEntries(pieceTypes.map((pieceType) => [
    pieceType,
    componentOptions(pieceType, current[pieceType].length),
  ])) as Record<PieceType, Coefficient[][]>;
  const options: PlayerComponents[] = [];

  for (const pawn of profiles.pawn) {
    for (const rook of profiles.rook) {
      for (const spy of profiles.spy) {
        for (const king of profiles.king) {
          options.push({
            pawn: pawn as PlayerComponents["pawn"],
            rook: rook as PlayerComponents["rook"],
            spy: spy as PlayerComponents["spy"],
            king: king as PlayerComponents["king"],
          });
        }
      }
    }
  }
  return options.sort((left, right) => componentDistance(left, current) - componentDistance(right, current));
}

export function findClosestPlayableConfiguration(player: Player, state: GameState): PlayableConfigurationHint | null {
  const rustHint = rustClosestPlayableConfiguration<PlayableConfigurationHint>(player, state);
  if (rustHint) return rustHint;
  const current = state.components[player];
  for (const components of allComponentOptions(state, player)) {
    const tuned = {
      ...state,
      components: {
        ...state.components,
        [player]: structuredClone(components),
      },
    };
    const field = evaluateField(tuned);
    const pieces = tuned.pieces.filter((piece) => piece.owner === player);
    for (const piece of pieces) {
      for (const destination of getLegalMoves(piece.id, tuned, field)) {
        const resolved = resolveOwnTurnConsequences(player, tuned, movePiece(tuned, piece.id, destination));
        if (!isKingUnprotected(player, resolved, evaluateField(resolved))) {
          return {
            components,
            pieceId: piece.id,
            pieceType: piece.type,
            destination,
            changedComponents: componentDistance(components, current),
          };
        }
      }
    }
  }
  return null;
}

export function beginTurn(state: GameState, options: RuleOptions = {}): GameState {
  const analyzeCheckmate = options.analyzeCheckmate ?? true;
  const rustState = rustBeginTurn(state, analyzeCheckmate);
  if (rustState) return rustState;
  if (state.status !== "playing") return state;
  const resolved = markInstability(state, evaluateField(state));
  const field = evaluateField(resolved);
  if (isKingUnprotected(state.currentPlayer, resolved, field)) {
    if (!analyzeCheckmate) {
      return { ...resolved, message: `${playerName(state.currentPlayer)} king is in check` };
    }
    const rescue = findClosestPlayableConfiguration(state.currentPlayer, resolved);
    if (rescue) {
      const rescueHint = rescue.changedComponents > 0
        ? `tune, then move ${rescue.pieceType} to ${boardCoordinate(rescue.destination)}`
        : `move ${rescue.pieceType} to ${boardCoordinate(rescue.destination)}`;
      return { ...resolved, message: `${playerName(state.currentPlayer)} king is in check · ${rescueHint}` };
    }
    return {
      ...resolved,
      status: winStatus(opponent(state.currentPlayer)),
      selectedPieceId: null,
      message: `${playerName(state.currentPlayer)} king is in check · no legal rescue found`,
    };
  }
  const unstable = getUnstablePieces(state.currentPlayer, resolved, field).filter((piece) => piece.type !== "king");
  if (unstable.length > 0) {
    return { ...resolved, message: `${playerName(state.currentPlayer)} must rescue an unstable ${unstable[0].type}` };
  }
  return { ...resolved, message: `${playerName(state.currentPlayer)} to move` };
}

function completeAction(previous: GameState, candidate: GameState, options: RuleOptions = {}): MoveResult {
  const selfResolved = resolveOwnTurnConsequences(previous.currentPlayer, previous, candidate);
  const selfField = evaluateField(selfResolved);
  if (isKingUnprotected(previous.currentPlayer, selfResolved, selfField)) {
    return { ok: false, state: previous, reason: "That move would leave your king unprotected." };
  }
  const enemy = opponent(previous.currentPlayer);
  const next = beginTurn({
    ...selfResolved,
    currentPlayer: enemy,
    turnNumber: previous.currentPlayer === "red" ? previous.turnNumber + 1 : previous.turnNumber,
    selectedPieceId: null,
    history: [...previous.history, snapshot(previous)],
  }, options);
  const losses = lostOwnPieces(previous.currentPlayer, previous, next);
  if (losses.length > 0 && next.status === "playing") {
    return { ok: true, state: { ...next, message: `${playerName(previous.currentPlayer)} lost ${losses.join(", ")} · ${next.message}` } };
  }
  return { ok: true, state: next };
}

export function applyMove(pieceId: string, destination: Position, state: GameState, options: RuleOptions = {}): MoveResult {
  const rustResult = rustApplyMove(pieceId, destination, state, options.analyzeCheckmate ?? true);
  if (rustResult) return rustResult;
  if (state.status !== "playing") return { ok: false, state, reason: "The game is over." };
  const field = evaluateField(state);
  const piece = state.pieces.find((candidate) => candidate.id === pieceId);
  if (!piece || piece.owner !== state.currentPlayer) return { ok: false, state, reason: "Choose one of your pieces." };

  if (!getLegalMoves(pieceId, state, field).some((move) => samePosition(move, destination))) {
    return { ok: false, state, reason: "That square is not a legal move." };
  }

  const candidate = {
    ...state,
    pieces: state.pieces.map((item) => item.id === pieceId ? { ...item, position: destination } : item),
  };
  return completeAction(state, candidate, options);
}

export function getPlayableMoves(pieceId: string, state: GameState, field: number[][] = evaluateField(state)): Position[] {
  const rustMoves = rustPlayableMoves(pieceId, state);
  if (rustMoves) return rustMoves;
  return getLegalMoves(pieceId, state, field).filter((destination) => applyMove(pieceId, destination, state).ok);
}

export function resignInCheck(state: GameState): MoveResult {
  const rustResult = rustResignInCheck(state);
  if (rustResult) return rustResult;
  if (state.status !== "playing") return { ok: false, state, reason: "The game is over." };
  const resolved = markInstability(state, evaluateField(state));
  if (!isKingUnprotected(state.currentPlayer, resolved, evaluateField(resolved))) {
    return { ok: false, state, reason: "You can resign only while your king is in check." };
  }
  return {
    ok: true,
    state: {
      ...resolved,
      status: winStatus(opponent(state.currentPlayer)),
      selectedPieceId: null,
      history: [...state.history, snapshot(state)],
      message: `${playerName(state.currentPlayer)} resigned while in check`,
    },
  };
}

export function applyClosestPlayableHint(state: GameState): MoveResult {
  const rustResult = rustApplyClosestPlayableHint(state);
  if (rustResult) return rustResult;
  if (state.status !== "playing") return { ok: false, state, reason: "The game is over." };
  const resolved = markInstability(state, evaluateField(state));
  if (!isKingUnprotected(state.currentPlayer, resolved, evaluateField(resolved))) {
    return { ok: false, state, reason: "Hints are available only while your king is in check." };
  }
  const hint = findClosestPlayableConfiguration(state.currentPlayer, resolved);
  if (!hint) return { ok: false, state, reason: "No legal escape exists." };

  const components = structuredClone(resolved.components);
  components[state.currentPlayer] = structuredClone(hint.components);
  const activationOrders = structuredClone(resolved.activationOrders);
  activationOrders[state.currentPlayer] = activationOrderForProfile(hint.components);
  const tuned = markInstability({ ...resolved, components }, evaluateField({ ...resolved, components }));
  const changeText = hint.changedComponents === 0
    ? "Current tuning works"
    : `${hint.changedComponents} control${hint.changedComponents === 1 ? "" : "s"} changed`;
  return {
    ok: true,
    state: {
      ...tuned,
      activationOrders,
      selectedPieceId: hint.pieceId,
      history: [...state.history, snapshot(state)],
      message: `Hint · ${changeText} · move ${hint.pieceType} to ${boardCoordinate(hint.destination)}`,
    },
  };
}

export function randomizeTuning(state: GameState, random: () => number = Math.random): MoveResult {
  const rolls = pieceTypes.map(() => random());
  const rustResult = rustRandomizeTuning(state, rolls);
  if (rustResult) return rustResult;
  if (state.status !== "playing") return { ok: false, state, reason: "The game is over." };
  const player = state.currentPlayer;
  const randomized = structuredClone(state.components[player]);
  for (const pieceType of pieceTypes) {
    const options = componentOptions(pieceType, randomized[pieceType].length);
    randomized[pieceType] = options[Math.floor(rolls[pieceTypes.indexOf(pieceType)] * options.length)] as never;
  }
  if (componentDistance(randomized, state.components[player]) === 0) {
    const alternatives = componentOptions("pawn", randomized.pawn.length)
      .filter((profile) => profile.some((value, index) => value !== randomized.pawn[index]));
    randomized.pawn = alternatives[0] as PlayerComponents["pawn"];
  }
  const components = structuredClone(state.components);
  components[player] = randomized;
  const activationOrders = structuredClone(state.activationOrders);
  activationOrders[player] = activationOrderForProfile(randomized);
  const candidate = { ...state, components, activationOrders };
  const marked = markInstability(candidate, evaluateField(candidate));
  const message = isKingUnprotected(player, marked, evaluateField(marked))
    ? `${playerName(player)} randomized tuning · king remains in check`
    : `${playerName(player)} randomized tuning · move a piece to end the turn`;
  return {
    ok: true,
    state: {
      ...marked,
      selectedPieceId: null,
      history: [...state.history, snapshot(state)],
      message,
    },
  };
}

export function resetTuning(state: GameState): MoveResult {
  const rustResult = rustResetTuning(state);
  if (rustResult) return rustResult;
  if (state.status !== "playing") return { ok: false, state, reason: "The game is over." };
  const player = state.currentPlayer;
  if (componentDistance(state.components[player], state.defaultComponents) === 0) {
    return { ok: false, state, reason: "Tuning already matches the defaults." };
  }
  const components = structuredClone(state.components);
  components[player] = structuredClone(state.defaultComponents);
  const activationOrders = structuredClone(state.activationOrders);
  activationOrders[player] = activationOrderForProfile(state.defaultComponents);
  const candidate = { ...state, components, activationOrders };
  const marked = markInstability(candidate, evaluateField(candidate));
  const message = isKingUnprotected(player, marked, evaluateField(marked))
    ? `${playerName(player)} reset tuning · king remains in check`
    : `${playerName(player)} reset tuning · move a piece to end the turn`;
  return {
    ok: true,
    state: {
      ...marked,
      selectedPieceId: null,
      history: [...state.history, snapshot(state)],
      message,
    },
  };
}

export function applyTuning(
  player: Player,
  pieceType: PieceType,
  componentIndex: number,
  value: Coefficient,
  state: GameState,
): MoveResult {
  const rustResult = rustApplyTuning(player, pieceType, componentIndex, value, state);
  if (rustResult) return rustResult;
  if (state.status !== "playing" || player !== state.currentPlayer) return { ok: false, state, reason: "It is not that player's turn." };
  if (value === 0) return { ok: false, state, reason: "Controls must stay at full strength." };
  const nextComponents = structuredClone(state.components);
  const activationOrders = structuredClone(state.activationOrders);
  const coefficients = nextComponents[player][pieceType];
  const currentValue = coefficients[componentIndex];
  if (currentValue === value) return { ok: false, state, reason: "Choose a different sign." };
  const activeIndices = coefficients.flatMap((coefficient, index) => coefficient === 0 ? [] : [index]);
  const existingOrder = activationOrders[player][pieceType]
    .filter((index) => activeIndices.includes(index));
  for (const index of activeIndices) {
    if (!existingOrder.includes(index)) existingOrder.push(index);
  }

  const wasActive = currentValue !== 0;
  const nextOrder = existingOrder.filter((index) => index !== componentIndex);
  if (!wasActive && activeIndices.length >= TUNING_STRENGTH[pieceType]) {
    const evictedIndex = nextOrder.shift();
    if (evictedIndex !== undefined) coefficients[evictedIndex] = 0;
  }
  coefficients[componentIndex] = value;
  nextOrder.push(componentIndex);
  activationOrders[player][pieceType] = nextOrder;

  const candidate = { ...state, components: nextComponents, activationOrders };
  const marked = markInstability(candidate, evaluateField(candidate));
  const message = isKingUnprotected(player, marked, evaluateField(marked))
    ? `${playerName(player)} king is in check · move to rescue the king`
    : `${playerName(player)} tuning · move a piece to end the turn`;

  return {
    ok: true,
    state: {
      ...marked,
      history: [...state.history, snapshot(state)],
      message,
    },
  };
}
