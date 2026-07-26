import { evaluateField } from "../field/evaluateField";
import { isSquareCompatible } from "../field/projection";
import type { GameState, Piece, Player } from "./types";

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

export function isKingUnprotected(player: Player, state: GameState, field: number[][]): boolean {
  const king = state.pieces.find((piece) => piece.owner === player && piece.type === "king");
  return Boolean(king && !isSquareCompatible(player, field[king.position.y][king.position.x]));
}

export function removeUnrescuedPieces(
  player: Player,
  state: GameState,
  rescueDeadlineIds: ReadonlySet<string>,
  field: number[][] = evaluateField(state),
): GameState {
  let next = state;
  let currentField = field;
  while (true) {
    const lostIds = new Set(
      getUnstablePieces(player, next, currentField)
        .filter((piece) => piece.type !== "king" && rescueDeadlineIds.has(piece.id))
        .map((piece) => piece.id),
    );
    if (lostIds.size === 0) return next;
    next = { ...next, pieces: next.pieces.filter((piece) => !lostIds.has(piece.id)) };
    currentField = evaluateField(next);
  }
}
