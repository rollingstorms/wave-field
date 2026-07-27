export const BOARD_SIZE = 7;
export const FIELD_EPSILON = 1e-9;
export const WAVE_DECAY_BASE = 2;
export const WAVE_ORIGIN_SCALE = 1;

export const PIECE_STRENGTH = {
  pawn: 1,
  rook: 2,
  spy: 1,
  king: 2,
} as const;

export const HOME_SQUARE_CONTRIBUTION = {
  pawn: 0.5,
  rook: 1,
  spy: 1.5,
  king: 0,
} as const;

export const COMPONENT_COUNTS = {
  pawn: 1,
  rook: 2,
  spy: 3,
  king: 3,
} as const;
