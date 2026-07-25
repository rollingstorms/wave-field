import { PIECE_STRENGTH } from "./constants";
import type { Coefficient, PieceType, PlayerComponents } from "./types";

export function getTuningLoad(coefficients: readonly Coefficient[]): number {
  return coefficients.filter((coefficient) => coefficient !== 0).length;
}

export function isTuningWithinStrength(pieceType: PieceType, coefficients: readonly Coefficient[]): boolean {
  return getTuningLoad(coefficients) <= PIECE_STRENGTH[pieceType];
}

export function canSetComponentValue(
  components: PlayerComponents,
  pieceType: PieceType,
  componentIndex: number,
  value: Coefficient,
): boolean {
  const next = [...components[pieceType]];
  next[componentIndex] = value;
  return isTuningWithinStrength(pieceType, next);
}
