import { describe, expect, it } from "vitest";
import { createInitialState } from "../game/initialState";
import { canSetComponentValue, getTuningLoad, isTuningWithinStrength } from "../game/tuning";

describe("component strength budget", () => {
  it("starts both players' rooks at positive-negative", () => {
    const state = createInitialState();

    expect(state.components.blue.rook).toEqual([1, -1]);
    expect(state.components.red.rook).toEqual([1, -1]);
  });

  it("counts positive and negative coefficients as active", () => {
    expect(getTuningLoad([1, 0, -1, 0])).toBe(2);
  });

  it("limits the combined number of positive and negative components", () => {
    expect(isTuningWithinStrength("king", [1, -1, 0])).toBe(true);
    expect(isTuningWithinStrength("king", [1, -1, 1])).toBe(false);
  });

  it("allows flipping an active component while at full strength", () => {
    const components = createInitialState().components.blue;
    components.king = [1, -1, 0];

    expect(canSetComponentValue(components, "king", 0, -1)).toBe(true);
    expect(canSetComponentValue(components, "king", 2, 1)).toBe(false);
    expect(canSetComponentValue(components, "king", 1, 0)).toBe(true);
  });
});
