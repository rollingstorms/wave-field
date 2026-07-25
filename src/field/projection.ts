import { FIELD_EPSILON } from "../game/constants";
import type { Player, Territory } from "../game/types";

export function projectFieldValue(value: number, epsilon = FIELD_EPSILON): Territory {
  if (value > epsilon) return "red";
  if (value < -epsilon) return "blue";
  return "neutral";
}

export function isSquareCompatible(player: Player, value: number, epsilon = FIELD_EPSILON): boolean {
  return player === "red" ? value >= -epsilon : value <= epsilon;
}
