import type { TypeFields } from "./evaluateField";
import { BOARD_SIZE, FIELD_EPSILON } from "../game/constants";
import type { PieceType } from "../game/types";

export const ENERGY_CHANNELS: Array<{ pieceType: PieceType; letter: string; channel: string }> = [
  { pieceType: "pawn", letter: "P", channel: "cyan" },
  { pieceType: "rook", letter: "R", channel: "magenta" },
  { pieceType: "spy", letter: "S", channel: "yellow" },
  { pieceType: "king", letter: "K", channel: "black" },
];

export type EnergyChannelState = Record<PieceType, boolean>;
export type EnergyValues = Record<PieceType, number>;

export interface CmykEnergyCell {
  raw: EnergyValues;
  ratios: EnergyValues;
  intensity: number;
  color: string;
}

export const ALL_ENERGY_CHANNELS: EnergyChannelState = {
  pawn: true,
  rook: true,
  spy: true,
  king: true,
};

function cmykColor(ratios: EnergyValues, intensity: number): string {
  const cyan = ratios.pawn * intensity;
  const magenta = ratios.rook * intensity;
  const yellow = ratios.spy * intensity;
  const black = ratios.king * intensity;
  const red = Math.round(255 * (1 - cyan) * (1 - black));
  const green = Math.round(255 * (1 - magenta) * (1 - black));
  const blue = Math.round(255 * (1 - yellow) * (1 - black));
  return `rgb(${red}, ${green}, ${blue})`;
}

export function createCmykEnergyGrid(typeFields: TypeFields, enabled: EnergyChannelState): CmykEnergyCell[][] {
  const totals = Array.from({ length: BOARD_SIZE }, (_, y) =>
    Array.from({ length: BOARD_SIZE }, (_, x) =>
      ENERGY_CHANNELS.reduce((sum, { pieceType }) =>
        sum + (enabled[pieceType] ? Math.abs(typeFields[pieceType][y][x]) : 0), 0),
    ),
  );
  const maximumTotal = Math.max(...totals.flat(), FIELD_EPSILON);

  return Array.from({ length: BOARD_SIZE }, (_, y) =>
    Array.from({ length: BOARD_SIZE }, (_, x) => {
      const raw = Object.fromEntries(
        ENERGY_CHANNELS.map(({ pieceType }) => [pieceType, typeFields[pieceType][y][x]]),
      ) as EnergyValues;
      const total = totals[y][x];
      const ratios = Object.fromEntries(
        ENERGY_CHANNELS.map(({ pieceType }) => [
          pieceType,
          enabled[pieceType] && total > FIELD_EPSILON ? Math.abs(raw[pieceType]) / total : 0,
        ]),
      ) as EnergyValues;
      const intensity = total / maximumTotal;
      return { raw, ratios, intensity, color: cmykColor(ratios, intensity) };
    }),
  );
}
