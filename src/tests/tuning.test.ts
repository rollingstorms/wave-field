import { describe, expect, it } from "vitest";
import { createInitialState } from "../game/initialState";
import { canSetComponentValue, getTuningLoad, isTuningAtStrength, isTuningWithinStrength } from "../game/tuning";
import { coefficientLabel } from "../components/ComponentControls";
import { randomizeTuning, resetTuning } from "../game/rules";

describe("component strength budget", () => {
  it("starts both players with the current default component set", () => {
    const state = createInitialState();

    expect(state.components.blue.pawn).toEqual([1]);
    expect(state.components.blue.rook).toEqual([1, 1]);
    expect(state.components.blue.spy).toEqual([1, 0]);
    expect(state.components.blue.king).toEqual([1, 1]);
    expect(state.components.red).toEqual(state.components.blue);
  });

  it("counts positive and negative coefficients as active", () => {
    expect(getTuningLoad([1, 0, -1, 0])).toBe(2);
  });

  it("limits the combined number of positive and negative components", () => {
    expect(isTuningWithinStrength("king", [1, -1])).toBe(true);
    expect(isTuningWithinStrength("king", [1, 1])).toBe(true);
    expect(isTuningWithinStrength("spy", [1, 0])).toBe(true);
    expect(isTuningWithinStrength("spy", [1, 1])).toBe(false);
  });

  it("recognizes exact full-strength tuning", () => {
    expect(isTuningAtStrength("king", [1, -1])).toBe(true);
    expect(isTuningAtStrength("king", [1, 0])).toBe(false);
    expect(isTuningAtStrength("spy", [1, 0])).toBe(true);
  });

  it("allows flipping an active component while at full strength", () => {
    const components = createInitialState().components.blue;
    components.king = [1, -1];

    expect(canSetComponentValue(components, "king", 0, -1)).toBe(true);
    expect(canSetComponentValue(components, "king", 1, 0)).toBe(true);
  });

  it("starts the king at the current two-active-slot tuning", () => {
    const state = createInitialState();

    expect(state.components.blue.king).toEqual([1, 1]);
    expect(state.components.red.king).toEqual([1, 1]);
    expect(canSetComponentValue(state.components.blue, "king", 0, -1)).toBe(true);
    expect(canSetComponentValue(state.components.blue, "king", 0, 0)).toBe(true);
  });

  it("displays component signs as field signs for each player", () => {
    expect(coefficientLabel("red", 1)).toBe("+");
    expect(coefficientLabel("red", -1)).toBe("-");
    expect(coefficientLabel("blue", 1)).toBe("-");
    expect(coefficientLabel("blue", -1)).toBe("+");
    expect(coefficientLabel("blue", 0)).toBe("0");
  });

  it("randomizes every type at full strength without ending the turn", () => {
    const state = createInitialState();
    const result = randomizeTuning(state, () => 0.75);

    expect(result.ok).toBe(true);
    expect(result.state.currentPlayer).toBe("blue");
    expect(result.state.history).toHaveLength(1);
    (["pawn", "rook", "spy", "king"] as const).forEach((pieceType) => {
      expect(isTuningAtStrength(pieceType, result.state.components.blue[pieceType])).toBe(true);
    });
    expect(result.state.components.red).toEqual(state.components.red);
  });

  it("forces at least one change when a roll matches the current profile", () => {
    const state = createInitialState();
    const rolls = [0, 0, 0, 0.27];
    const result = randomizeTuning(state, () => rolls.shift() ?? 0);

    expect(result.ok).toBe(true);
    expect(result.state.components.blue).not.toEqual(state.components.blue);
  });

  it("resets only the current player's tuning to the configured defaults", () => {
    const state = createInitialState();
    state.components.blue.pawn = [0];
    const result = resetTuning(state);

    expect(result.ok).toBe(true);
    expect(result.state.components.blue).toEqual(state.defaultComponents);
    expect(result.state.components.red).toEqual(state.components.red);
    expect(result.state.currentPlayer).toBe("blue");
    expect(result.state.history).toHaveLength(1);
  });

  it("does not add history when tuning already matches the defaults", () => {
    const result = resetTuning(createInitialState());

    expect(result.ok).toBe(false);
    expect(result.reason).toContain("already matches");
    expect(result.state.history).toHaveLength(0);
  });
});
