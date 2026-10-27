import type { Coefficient, PieceType, Position } from "../game/types";

const WIDE_SUPPORT = 1;
const FOCUSED_SUPPORT = 2;
const HOME_SUPPORT: Record<PieceType, number> = { pawn: 1, spy: 1, rook: 2, king: 2 };

function onAxis(x: number, y: number): boolean {
  return (x === 0) !== (y === 0);
}

function onDiagonal(x: number, y: number): boolean {
  return x !== 0 && Math.abs(x) === Math.abs(y);
}

function profileKey(coefficients: readonly Coefficient[]): string {
  return coefficients.map((value) => value > 0 ? "+" : value < 0 ? "-" : "0").join("");
}

export function sparsePatternDescription(pieceType: PieceType, coefficients: readonly Coefficient[]): string {
  const key = profileKey(coefficients);
  if (pieceType === "pawn") return key === "+" ? "Wide cross · 1" : key === "-" ? "Focused diagonals · 2" : "No outer support";
  if (pieceType === "spy") return key === "+" ? "Wide diamond · 1" : key === "-" ? "Focused diamond · 2" : "No outer support";
  if (pieceType === "rook") {
    if (key === "++") return "Wide rays · 1";
    if (key === "+-") return "Focused cross · 2";
    if (key === "-+") return "Focused diagonals · 2";
    if (key === "--") return "Tight cross · 2";
  }
  if (pieceType === "king") {
    if (key === "++") return "Wide square · 1";
    if (key === "+-") return "Focused diamond · 2";
    if (key === "-+") return "Focused diagonals · 2";
    if (key === "--") return "Tight cross · 2";
  }
  return "Partial pattern";
}

export function evaluateSparsePattern(pieceType: PieceType, coefficients: readonly Coefficient[], delta: Position): number {
  const { x, y } = delta;
  if (x === 0 && y === 0) return HOME_SUPPORT[pieceType];
  const radius = Math.max(Math.abs(x), Math.abs(y));
  if (radius > 2) return 0;
  const axis = onAxis(x, y);
  const diagonal = onDiagonal(x, y);
  const diamond = Math.abs(x) + Math.abs(y) <= 2;
  const key = profileKey(coefficients);

  if (pieceType === "pawn") {
    if (key === "+") return axis ? WIDE_SUPPORT : 0;
    if (key === "-") return diagonal && radius === 1 ? FOCUSED_SUPPORT : 0;
    return 0;
  }
  if (pieceType === "spy") {
    if (key === "+") return diamond ? WIDE_SUPPORT : 0;
    if (key === "-") return axis && radius === 1 ? FOCUSED_SUPPORT : 0;
    return 0;
  }
  if (key === "++") {
    if (pieceType === "rook") return axis || diagonal ? WIDE_SUPPORT : 0;
    return WIDE_SUPPORT;
  }
  if (key === "+-") {
    if (pieceType === "rook") return axis ? FOCUSED_SUPPORT : 0;
    return diamond ? FOCUSED_SUPPORT : 0;
  }
  if (key === "-+") return diagonal ? FOCUSED_SUPPORT : 0;
  if (key === "--") return axis && radius === 1 ? FOCUSED_SUPPORT : 0;

  // Developer profiles with a disabled component retain one visible stencil.
  if (coefficients[0] !== 0) return axis ? WIDE_SUPPORT : 0;
  if (coefficients[1] !== 0) return diagonal ? WIDE_SUPPORT : 0;
  return 0;
}
