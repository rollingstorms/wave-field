import { BOARD_SIZE } from "./constants";
import type { GameState, Piece, Position } from "./types";
import { isSquareCompatible } from "../field/projection";

export function inBounds(position: Position): boolean {
  return position.x >= 0 && position.x < BOARD_SIZE && position.y >= 0 && position.y < BOARD_SIZE;
}

export function samePosition(a: Position, b: Position): boolean {
  return a.x === b.x && a.y === b.y;
}

export function getPieceAt(state: GameState, position: Position): Piece | undefined {
  return state.pieces.find((piece) => samePosition(piece.position, position));
}

export function isAdjacent(a: Position, b: Position): boolean {
  const dx = Math.abs(a.x - b.x);
  const dy = Math.abs(a.y - b.y);
  return dx <= 1 && dy <= 1 && dx + dy > 0;
}

export function canPieceEnter(piece: Piece, destination: Position, state: GameState, field: number[][]): boolean {
  if (!inBounds(destination) || !isAdjacent(piece.position, destination)) return false;
  if (getPieceAt(state, destination)) return false;
  return piece.type === "spy" || isSquareCompatible(piece.owner, field[destination.y][destination.x]);
}

export function getLegalMoves(pieceId: string, state: GameState, field: number[][]): Position[] {
  const piece = state.pieces.find((candidate) => candidate.id === pieceId);
  if (!piece) return [];
  const moves: Position[] = [];
  for (let dy = -1; dy <= 1; dy += 1) {
    for (let dx = -1; dx <= 1; dx += 1) {
      const destination = { x: piece.position.x + dx, y: piece.position.y + dy };
      if (canPieceEnter(piece, destination, state, field)) moves.push(destination);
    }
  }
  return moves;
}
