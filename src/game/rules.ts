import { evaluateField } from "../field/evaluateField";
import { BOARD_SIZE, PIECE_STRENGTH } from "./constants";
import type { Coefficient, GameState, MoveResult, PieceType, Player, PlayerComponents, Position } from "./types";
import { getLegalMoves, samePosition } from "./movement";
import { canSetComponentValue, isTuningWithinStrength } from "./tuning";
import { getUnstablePieces, isKingUnprotected, markInstability, removeUnrescuedPieces } from "./victory";
import { snapshot } from "./initialState";

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
  return `${position.x + 1},${BOARD_SIZE - position.y}`;
}

const pieceTypes: PieceType[] = ["pawn", "rook", "spy", "king"];
const coefficientValues: Coefficient[] = [1, 0, -1];

interface KingRescue {
  pieceType: PieceType;
  destination: Position;
  requiresTuning: boolean;
}

interface RuleOptions {
  analyzeCheckmate?: boolean;
}

function componentOptions(pieceType: PieceType, count: number): Coefficient[][] {
  const options: Coefficient[][] = [];
  function build(values: Coefficient[]) {
    if (values.length === count) {
      if (isTuningWithinStrength(pieceType, values)) options.push(values);
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
  const rescueDeadlineIds = new Set(
    getUnstablePieces(player, previous, evaluateField(previous))
      .filter((piece) => piece.type !== "king")
      .map((piece) => piece.id),
  );
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

function rescueComponentOptions(state: GameState, player: Player): PlayerComponents[] {
  const current = state.components[player];
  const options = [structuredClone(current)];
  const seen = new Set([JSON.stringify(current)]);

  for (const pieceType of pieceTypes) {
    for (const profile of componentOptions(pieceType, current[pieceType].length)) {
      const next = structuredClone(current);
      next[pieceType] = profile as never;
      const key = JSON.stringify(next);
      if (!seen.has(key)) {
        seen.add(key);
        options.push(next);
      }
    }
  }
  return options;
}

function findKingRescue(player: Player, state: GameState): KingRescue | null {
  for (const components of rescueComponentOptions(state, player)) {
    const requiresTuning = JSON.stringify(state.components[player]) !== JSON.stringify(components);
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
          return { pieceType: piece.type, destination, requiresTuning };
        }
      }
    }
  }
  return null;
}

export function beginTurn(state: GameState, options: RuleOptions = {}): GameState {
  const analyzeCheckmate = options.analyzeCheckmate ?? true;
  if (state.status !== "playing") return state;
  const resolved = markInstability(state, evaluateField(state));
  const field = evaluateField(resolved);
  if (isKingUnprotected(state.currentPlayer, resolved, field)) {
    if (!analyzeCheckmate) {
      return { ...resolved, message: `${playerName(state.currentPlayer)} king is in check` };
    }
    const rescue = findKingRescue(state.currentPlayer, resolved);
    if (rescue) {
      const rescueHint = rescue.requiresTuning
        ? `tune, then move ${rescue.pieceType} to ${boardCoordinate(rescue.destination)}`
        : `move ${rescue.pieceType} to ${boardCoordinate(rescue.destination)}`;
      return { ...resolved, message: `${playerName(state.currentPlayer)} king is in check · ${rescueHint}` };
    }
    return { ...resolved, status: winStatus(opponent(state.currentPlayer)), message: `${playerName(state.currentPlayer)} king is checkmated` };
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
  return getLegalMoves(pieceId, state, field).filter((destination) => applyMove(pieceId, destination, state).ok);
}

export function applyTuning(
  player: Player,
  pieceType: PieceType,
  componentIndex: number,
  value: Coefficient,
  state: GameState,
): MoveResult {
  if (state.status !== "playing" || player !== state.currentPlayer) return { ok: false, state, reason: "It is not that player's turn." };
  if (state.components[player][pieceType][componentIndex] === value) return { ok: false, state, reason: "Choose a different coefficient." };
  if (!canSetComponentValue(state.components[player], pieceType, componentIndex, value)) {
    const strength = PIECE_STRENGTH[pieceType];
    return { ok: false, state, reason: `${pieceType[0].toUpperCase()}${pieceType.slice(1)} strength allows up to ${strength} active component${strength === 1 ? "" : "s"}.` };
  }

  const nextComponents = structuredClone(state.components);
  nextComponents[player][pieceType][componentIndex] = value;
  const candidate = { ...state, components: nextComponents };
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
