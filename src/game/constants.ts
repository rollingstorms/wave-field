export const BOARD_SIZE = 7;
export const FIELD_EPSILON = 1e-9;

export const PIECE_STRENGTH = {
  pawn: 1,
  rook: 2,
  spy: 1,
  king: 2,
} as const;

export const COMPONENT_COUNTS = {
  pawn: 1,
  rook: 2,
  spy: 3,
  king: 4,
} as const;
