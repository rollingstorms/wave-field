import { evaluateField } from "../field/evaluateField";
import { isSquareCompatible } from "../field/projection";
import type { ComponentDefinitions, GameState, Piece, Player } from "./types";
import { getLegalMoves } from "./movement";

export function getUnstablePieces(player: Player, state: GameState, field: number[][]): Piece[] {
  return state.pieces.filter((piece) => {
    if (piece.owner !== player || piece.type === "spy") return false;
    return !isSquareCompatible(player, field[piece.position.y][piece.position.x]);
  });
}

export function markInstability(state: GameState, field: number[][]): GameState {
  return {
    ...state,
    pieces: state.pieces.map((piece) => ({
      ...piece,
      unstable: piece.type !== "spy" && !isSquareCompatible(piece.owner, field[piece.position.y][piece.position.x]),
    })),
  };
}

export function isKingTrapped(player: Player, state: GameState, field: number[][]): boolean {
  const king = state.pieces.find((piece) => piece.owner === player && piece.type === "king");
  if (!king || isSquareCompatible(player, field[king.position.y][king.position.x])) return false;
  return getLegalMoves(king.id, state, field).length === 0;
}

export function removeUnrescuedPieces(
  player: Player,
  state: GameState,
  rescueDeadlineIds: ReadonlySet<string>,
  field: number[][] = evaluateField(state),
): GameState {
  const lostIds = new Set(
    getUnstablePieces(player, state, field)
      .filter((piece) => piece.type !== "king" && rescueDeadlineIds.has(piece.id))
      .map((piece) => piece.id),
  );
  if (lostIds.size === 0) return state;
  return {
    ...state,
    pieces: state.pieces.filter((piece) => !lostIds.has(piece.id)),
  };
}

export function resolveForcedRemovals(
  player: Player,
  state: GameState,
  definitions: ComponentDefinitions = state.definitions,
): GameState {
  let next = state;
  let changed = true;
  while (changed) {
    changed = false;
    const field = evaluateField(next, definitions);
    const stranded = getUnstablePieces(player, next, field).filter((piece) => piece.type !== "king" && getLegalMoves(piece.id, next, field).length === 0);
    if (stranded.length > 0) {
      const ids = new Set(stranded.map((piece) => piece.id));
      next = { ...next, pieces: next.pieces.filter((piece) => !ids.has(piece.id)) };
      changed = true;
    }
  }
  return markInstability(next, evaluateField(next, definitions));
}
