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

function isSpectrallyPassable(piece: Piece, position: Position, field: number[][]): boolean {
  return piece.type === "spy" || isSquareCompatible(piece.owner, field[position.y][position.x]);
}

function movementStep(origin: Position, destination: Position): Position | null {
  const dx = destination.x - origin.x;
  const dy = destination.y - origin.y;
  if (dx === 0 && dy === 0) return null;
  if (dx !== 0 && dy !== 0 && Math.abs(dx) !== Math.abs(dy)) return null;
  return { x: Math.sign(dx), y: Math.sign(dy) };
}

export function canPieceEnter(piece: Piece, destination: Position, state: GameState, field: number[][]): boolean {
  if (!inBounds(destination)) return false;
  const step = movementStep(piece.position, destination);
  if (!step) return false;

  let position = { x: piece.position.x + step.x, y: piece.position.y + step.y };
  while (inBounds(position)) {
    if (getPieceAt(state, position) || !isSpectrallyPassable(piece, position, field)) return false;
    if (samePosition(position, destination)) return true;
    position = { x: position.x + step.x, y: position.y + step.y };
  }
  return false;
}

export function getLegalMoves(pieceId: string, state: GameState, field: number[][]): Position[] {
  const piece = state.pieces.find((candidate) => candidate.id === pieceId);
  if (!piece) return [];
  const moves: Position[] = [];
  for (let dy = -1; dy <= 1; dy += 1) {
    for (let dx = -1; dx <= 1; dx += 1) {
      if (dx === 0 && dy === 0) continue;
      let destination = { x: piece.position.x + dx, y: piece.position.y + dy };
      while (inBounds(destination)) {
        if (getPieceAt(state, destination) || !isSpectrallyPassable(piece, destination, field)) break;
        moves.push(destination);
        destination = { x: destination.x + dx, y: destination.y + dy };
      }
    }
  }
  return moves;
}
