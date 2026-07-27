export const BOARD_SIZE = 7;
export const FIELD_EPSILON = 1e-9;
export const WAVE_DECAY_BASE = 2;
export const WAVE_ORIGIN_SCALE = 1;

export const PIECE_STRENGTH = {
  pawn: 1,
  rook: 2,
  spy: 2,
  king: 2,
} as const;

export const TUNING_STRENGTH = {
  pawn: 1,
  rook: 2,
  spy: 1,
  king: 2,
} as const;

export const DEFAULT_HOME_ENERGY = {
  pawn: 0,
  rook: 0,
  spy: 0.5,
  king: 0,
} as const;

export const DEFAULT_WAVE_SCALES = {
  pawn: { friendly: 3, hostile: 1 },
  rook: { friendly: 3, hostile: 1 },
  spy: { friendly: 3, hostile: 0 },
  king: { friendly: 3, hostile: 2 },
} as const;

export const COMPONENT_COUNTS = {
  pawn: 1,
  rook: 2,
  spy: 3,
  king: 3,
} as const;
