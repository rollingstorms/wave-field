import { describe, expect, it } from "vitest";
import { evaluateField } from "../field/evaluateField";
import { createInitialState } from "../game/initialState";

describe("initial symmetry", () => {
  it("keeps the initial field nearly balanced with the asymmetric tower grid", () => {
    const field = evaluateField(createInitialState());
    let sum = 0;
    let maxPairDelta = 0;
    for (let y = 0; y < 7; y += 1) {
      for (let x = 0; x < 7; x += 1) {
        maxPairDelta = Math.max(maxPairDelta, Math.abs(field[y][x] + field[6 - y][6 - x]));
        sum += field[y][x];
      }
    }
    expect(field[3][3]).toBeCloseTo(0);
    expect(sum).toBeCloseTo(0.75);
    expect(maxPairDelta).toBeCloseTo(0.75);
  });
});
