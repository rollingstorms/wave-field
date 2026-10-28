import type { Coefficient, PieceType, Position } from "../game/types";
import { SPARSE_HOME_ENERGY } from "../game/constants";

const WIDE_SUPPORT = 1;
const FOCUSED_SUPPORT = 2;
const TIGHT_SUPPORT = 4;

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
  if (pieceType === "spy") return key === "+" ? "Wide diamond · 1" : key === "-" ? "Focused diamond · 3" : "No outer support";
  if (pieceType === "rook") {
    if (key === "++") return "Wide rays · 1";
    if (key === "+-") return "Focused cross · 2";
    if (key === "-+") return "Focused diagonals · 2";
    if (key === "--") return "Tight cross · 4";
  }
  if (pieceType === "king") {
    if (key === "++") return "Wide square · 1";
    if (key === "+-") return "Focused diamond · 2";
    if (key === "-+") return "Focused diagonals · 3";
    if (key === "--") return "Tight cross · 6";
  }
  return "Partial pattern";
}

function supportAtOffset(pieceType: PieceType, coefficients: readonly Coefficient[], delta: Position): number {
  const { x, y } = delta;
  if (x === 0 && y === 0) return SPARSE_HOME_ENERGY[pieceType];
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
    if (key === "-") return axis && radius === 1 ? 3 : 0;
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
  if (key === "-+") return diagonal ? (pieceType === "king" ? 3 : FOCUSED_SUPPORT) : 0;
  if (key === "--") return axis && radius === 1 ? (pieceType === "king" ? 6 : TIGHT_SUPPORT) : 0;

  // Developer profiles with a disabled component retain one visible stencil.
  if (coefficients[0] !== 0) return axis ? WIDE_SUPPORT : 0;
  if (coefficients[1] !== 0) return diagonal ? WIDE_SUPPORT : 0;
  return 0;
}

export interface SparsePatternCell extends Position {
  support: number;
}

interface SparsePattern {
  cells: readonly SparsePatternCell[];
  values: readonly number[];
}

const patterns = new Map<string, SparsePattern>();

function patternFor(pieceType: PieceType, coefficients: readonly Coefficient[]): SparsePattern {
  const key = `${pieceType}:${profileKey(coefficients)}`;
  const cached = patterns.get(key);
  if (cached) return cached;
  const cells: SparsePatternCell[] = [];
  const values: number[] = [];
  for (let y = -2; y <= 2; y += 1) {
    for (let x = -2; x <= 2; x += 1) {
      const support = supportAtOffset(pieceType, coefficients, { x, y });
      values.push(support);
      if (support > 0) cells.push({ x, y, support });
    }
  }
  const pattern = { cells, values };
  patterns.set(key, pattern);
  return pattern;
}

export function sparsePatternCells(pieceType: PieceType, coefficients: readonly Coefficient[]): readonly SparsePatternCell[] {
  return patternFor(pieceType, coefficients).cells;
}

export function evaluateSparsePattern(pieceType: PieceType, coefficients: readonly Coefficient[], delta: Position): number {
  if (Math.abs(delta.x) > 2 || Math.abs(delta.y) > 2) return 0;
  return patternFor(pieceType, coefficients).values[(delta.y + 2) * 5 + delta.x + 2] ?? 0;
}
