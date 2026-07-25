import { evaluateField } from "../field/evaluateField";
import { PIECE_STRENGTH } from "./constants";
import type { Coefficient, GameState, MoveResult, PieceType, Player, Position } from "./types";
import { getLegalMoves, samePosition } from "./movement";
import { canSetComponentValue } from "./tuning";
import { getUnstablePieces, isKingTrapped, markInstability, resolveForcedRemovals } from "./victory";
import { snapshot } from "./initialState";

export function opponent(player: Player): Player {
  return player === "red" ? "blue" : "red";
}

function winStatus(winner: Player) {
  return winner === "red" ? "red-won" as const : "blue-won" as const;
}

export function beginTurn(state: GameState): GameState {
  if (state.status !== "playing") return state;
  const resolved = resolveForcedRemovals(state.currentPlayer, state);
  const field = evaluateField(resolved);
  if (isKingTrapped(state.currentPlayer, resolved, field)) {
    return { ...resolved, status: winStatus(opponent(state.currentPlayer)), message: `${state.currentPlayer === "red" ? "Red" : "Blue"} king is trapped` };
  }
  const unstable = getUnstablePieces(state.currentPlayer, resolved, field).filter((piece) => piece.type !== "king");
  if (unstable.length > 0) {
    return { ...resolved, message: `${state.currentPlayer === "red" ? "Red" : "Blue"} must rescue an unstable ${unstable[0].type}` };
  }
  return { ...resolved, message: `${state.currentPlayer === "red" ? "Red" : "Blue"} to move` };
}

function completeAction(previous: GameState, candidate: GameState): MoveResult {
  const marked = markInstability(candidate, evaluateField(candidate));
  const selfResolved = resolveForcedRemovals(previous.currentPlayer, marked);
  const selfField = evaluateField(selfResolved);
  if (isKingTrapped(previous.currentPlayer, selfResolved, selfField)) {
    return { ok: false, state: previous, reason: "That action would trap your own king." };
  }
  const enemy = opponent(previous.currentPlayer);
  if (isKingTrapped(enemy, selfResolved, selfField)) {
    return {
      ok: true,
      state: { ...selfResolved, status: winStatus(previous.currentPlayer), history: [...previous.history, snapshot(previous)], selectedPieceId: null, message: `${enemy === "red" ? "Red" : "Blue"} king is trapped` },
    };
  }
  const next = beginTurn({
    ...selfResolved,
    currentPlayer: enemy,
    turnNumber: previous.currentPlayer === "red" ? previous.turnNumber + 1 : previous.turnNumber,
    selectedPieceId: null,
    history: [...previous.history, snapshot(previous)],
  });
  return { ok: true, state: next };
}

export function applyMove(pieceId: string, destination: Position, state: GameState): MoveResult {
  if (state.status !== "playing") return { ok: false, state, reason: "The game is over." };
  const field = evaluateField(state);
  const piece = state.pieces.find((candidate) => candidate.id === pieceId);
  if (!piece || piece.owner !== state.currentPlayer) return { ok: false, state, reason: "Choose one of your pieces." };

  const unstable = getUnstablePieces(state.currentPlayer, state, field).filter((candidate) => candidate.type !== "king");
  if (unstable.length > 0 && !unstable.some((candidate) => candidate.id === pieceId)) {
    return { ok: false, state, reason: "You must rescue an unstable piece." };
  }
  if (!getLegalMoves(pieceId, state, field).some((move) => samePosition(move, destination))) {
    return { ok: false, state, reason: "That square is not a legal move." };
  }

  const candidate = {
    ...state,
    pieces: state.pieces.map((item) => item.id === pieceId ? { ...item, position: destination } : item),
  };
  return completeAction(state, candidate);
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
  if (isKingTrapped(player, marked, evaluateField(marked))) {
    return { ok: false, state, reason: "That tuning would trap your own king." };
  }

  return {
    ok: true,
    state: {
      ...marked,
      history: [...state.history, snapshot(state)],
      message: `${player === "red" ? "Red" : "Blue"} tuning · move a piece to end the turn`,
    },
  };
}
