const routePath = globalThis.location?.pathname.replace(/\/$/, "") ?? "";

export const STANDARD_BOARD_SIZE = 7;
export const BIG_BOARD_SIZE = 15;
export const BOARD_SIZE = routePath.endsWith("/big") ? BIG_BOARD_SIZE : STANDARD_BOARD_SIZE;
export const BOARD_CENTER = Math.floor(BOARD_SIZE / 2);
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

export const DEFAULT_COMPONENT_COUNTS = {
  pawn: 1,
  rook: 2,
  spy: 2,
  king: 2,
} as const;

export const DEBUG_COMPONENT_COUNT_LIMITS = {
  pawn: 1,
  rook: 2,
  spy: 3,
  king: 3,
} as const;

export const DEFAULT_HOME_ENERGY = {
  pawn: 0,
  rook: 0,
  spy: 0.5,
  king: 0,
} as const;

export const DEFAULT_WAVE_SCALES = {
  pawn: { friendly: 4, hostile: 1 },
  rook: { friendly: 3, hostile: 1 },
  spy: { friendly: 3, hostile: 0 },
  king: { friendly: 4, hostile: 2 },
} as const;

export function tuningStrengthFor(pieceType: keyof typeof TUNING_STRENGTH, componentCount: number): number {
  return Math.min(TUNING_STRENGTH[pieceType], componentCount);
}
