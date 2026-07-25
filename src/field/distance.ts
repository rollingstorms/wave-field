import type { Position } from "../game/types";

export function chebyshevDistance(source: Position, target: Position): number {
  return Math.max(Math.abs(source.x - target.x), Math.abs(source.y - target.y));
}

export function offset(source: Position, target: Position): Position {
  return { x: target.x - source.x, y: target.y - source.y };
}
