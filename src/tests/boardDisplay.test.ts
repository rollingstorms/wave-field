import { describe, expect, it } from "vitest";
import { BOARD_SIZE } from "../game/constants";
import { sampleBoardDisplay, sampleBoardField } from "../field/boardDisplay";
import { boardDisplayColor } from "../field/continuousColor";

describe("iOS-style board field display", () => {
  it("keeps a uniform game field uniform at edges and between squares", () => {
    const field = Array.from({ length: BOARD_SIZE }, () => Array(BOARD_SIZE).fill(2));
    const samples = sampleBoardDisplay(field);
    expect(samples).toHaveLength(BOARD_SIZE * 9);
    expect(samples[0]).toHaveLength(BOARD_SIZE * 9);
    expect(sampleBoardField(field, 0.25, 2.5).value).toBeCloseTo(2);
    expect(sampleBoardField(field, 0.25, 2.5).activity).toBeCloseTo(2);
  });

  it("preserves activity at a cancellation boundary", () => {
    const field = Array.from({ length: BOARD_SIZE }, (_, y) =>
      Array.from({ length: BOARD_SIZE }, (_, x) => x < 3 ? -2 : 2),
    );
    const sample = sampleBoardField(field, 2.5, 3);
    expect(Math.abs(sample.value)).toBeLessThan(2);
    expect(sample.activity).toBeCloseTo(2);
  });

  it("uses a fixed color scale across field changes", () => {
    expect(boardDisplayColor(0, 0)).toBe("rgb(19, 20, 22)");
    expect(boardDisplayColor(2.4, 1)).toBe("rgb(226, 74, 59)");
    expect(boardDisplayColor(0, 0)).not.toBe(boardDisplayColor(0, 1));
  });
});
