import { describe, expect, it } from "vitest";
import { ALL_ENERGY_CHANNELS, createCmykEnergyGrid } from "../field/cmykEnergy";
import type { TypeFields } from "../field/evaluateField";
import type { PieceType } from "../game/types";

function fields(values: Partial<Record<PieceType, number>>): TypeFields {
  return Object.fromEntries((["pawn", "rook", "spy", "king"] as PieceType[]).map((pieceType) => [
    pieceType,
    Array.from({ length: 7 }, () => Array.from({ length: 7 }, () => values[pieceType] ?? 0)),
  ])) as TypeFields;
}

describe("CMYK energy view", () => {
  it("maps isolated pawn energy to cyan while preserving intensity", () => {
    const grid = createCmykEnergyGrid(fields({ pawn: -2 }), ALL_ENERGY_CHANNELS);
    const cell = grid[0][0];

    expect(cell.raw.pawn).toBe(-2);
    expect(cell.ratios).toEqual({ pawn: 1, rook: 0, spy: 0, king: 0 });
    expect(cell.intensity).toBe(1);
    expect(cell.color).toBe("rgb(0, 255, 255)");
  });

  it("uses absolute signed sums for local composition ratios", () => {
    const cell = createCmykEnergyGrid(
      fields({ pawn: 1, rook: -2, spy: 1, king: 0 }),
      ALL_ENERGY_CHANNELS,
    )[0][0];

    expect(cell.ratios.pawn).toBeCloseTo(0.25);
    expect(cell.ratios.rook).toBeCloseTo(0.5);
    expect(cell.ratios.spy).toBeCloseTo(0.25);
    expect(cell.raw.rook).toBe(-2);
  });

  it("removes disabled types from the rendered mix", () => {
    const cell = createCmykEnergyGrid(
      fields({ pawn: 1, rook: 3 }),
      { pawn: true, rook: false, spy: false, king: false },
    )[0][0];

    expect(cell.ratios).toEqual({ pawn: 1, rook: 0, spy: 0, king: 0 });
    expect(cell.color).toBe("rgb(0, 255, 255)");
  });

  it("renders white zero-energy cells when every channel is off", () => {
    const cell = createCmykEnergyGrid(
      fields({ pawn: 1, rook: 1, spy: 1, king: 1 }),
      { pawn: false, rook: false, spy: false, king: false },
    )[0][0];

    expect(cell.intensity).toBe(0);
    expect(cell.ratios).toEqual({ pawn: 0, rook: 0, spy: 0, king: 0 });
    expect(cell.color).toBe("rgb(255, 255, 255)");
  });
});
