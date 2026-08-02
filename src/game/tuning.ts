import { tuningStrengthFor } from "./constants";
import type { ActivationOrders, Coefficient, PieceType, Player, PlayerActivationOrder, PlayerComponents } from "./types";

const pieceTypes: PieceType[] = ["pawn", "rook", "spy", "king"];

export function getTuningLoad(coefficients: readonly Coefficient[]): number {
  return coefficients.filter((coefficient) => coefficient !== 0).length;
}

export function isTuningWithinStrength(pieceType: PieceType, coefficients: readonly Coefficient[]): boolean {
  return getTuningLoad(coefficients) <= tuningStrengthFor(pieceType, coefficients.length);
}

export function isTuningAtStrength(pieceType: PieceType, coefficients: readonly Coefficient[]): boolean {
  return getTuningLoad(coefficients) === tuningStrengthFor(pieceType, coefficients.length);
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

export function activationOrderForProfile(components: PlayerComponents): PlayerActivationOrder {
  return Object.fromEntries(pieceTypes.map((pieceType) => [
    pieceType,
    components[pieceType].flatMap((coefficient, index) => coefficient === 0 ? [] : [index]),
  ])) as PlayerActivationOrder;
}

export function activationOrdersForPlayers(components: Record<Player, PlayerComponents>): ActivationOrders {
  return {
    blue: activationOrderForProfile(components.blue),
    red: activationOrderForProfile(components.red),
  };
}
