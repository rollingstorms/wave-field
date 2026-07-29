import type { PieceType } from "./types";

export const PIECE_TYPES: PieceType[] = ["pawn", "rook", "spy", "king"];

export const PIECE_DISPLAY_NAMES: Record<PieceType, string> = {
  pawn: "Round Hat",
  rook: "Tower",
  spy: "Triangle Hat",
  king: "Big Hat",
};

export const PIECE_DISPLAY_NAMES_PLURAL: Record<PieceType, string> = {
  pawn: "Round Hats",
  rook: "Towers",
  spy: "Triangle Hats",
  king: "Big Hats",
};

export const PIECE_INITIALS: Record<PieceType, string> = {
  pawn: "R",
  rook: "T",
  spy: "A",
  king: "B",
};

export function pieceName(pieceType: PieceType) {
  return PIECE_DISPLAY_NAMES[pieceType];
}

export function pieceNamePlural(pieceType: PieceType) {
  return PIECE_DISPLAY_NAMES_PLURAL[pieceType];
}

export function pieceNameLower(pieceType: PieceType) {
  return pieceName(pieceType).toLowerCase();
}
