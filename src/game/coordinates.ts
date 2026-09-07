import { BOARD_SIZE } from "./constants";
import type { PrecisePosition } from "./types";

function isIntegerCoordinate(value: number): boolean {
  return Number.isInteger(value);
}

function trimFixed(value: number): string {
  return value.toFixed(3).replace(/\.?0+$/, "");
}

export function boardCoordinate(position: PrecisePosition): string {
  if (isIntegerCoordinate(position.x) && isIntegerCoordinate(position.y)) {
    return `${String.fromCharCode(65 + position.x)}${BOARD_SIZE - position.y}`;
  }
  return `(${trimFixed(position.x)}, ${trimFixed(position.y)})`;
}
