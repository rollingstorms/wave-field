import { BOARD_SIZE } from "./constants";
import type { Position } from "./types";

const THIRD_SQUARE_INSET = 2;

function dedupePositions(positions: Position[]): Position[] {
  const seen = new Set<string>();
  return positions.filter((position) => {
    const key = `${position.x}:${position.y}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export function createAmpSquares(boardSize = BOARD_SIZE): Position[] {
  const low = THIRD_SQUARE_INSET;
  const high = boardSize - THIRD_SQUARE_INSET - 1;
  if (low < 0 || high < 0 || low > high) return [];
  return dedupePositions([
    { x: low, y: low },
    { x: high, y: low },
    { x: low, y: high },
    { x: high, y: high },
  ]);
}

export function isAmpSquare(position: Position, ampSquares: Position[] = []): boolean {
  return ampSquares.some((square) => square.x === position.x && square.y === position.y);
}
