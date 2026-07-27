import { describe, expect, it } from "vitest";
import { continuousFieldColor } from "../field/continuousColor";

describe("continuous field color", () => {
  it("renders zero as neutral", () => {
    expect(continuousFieldColor(0, 8)).toBe("rgb(184, 181, 173)");
  });

  it("renders maximum positive and negative values at their territory colors", () => {
    expect(continuousFieldColor(8, 8)).toBe("rgb(200, 75, 64)");
    expect(continuousFieldColor(-8, 8)).toBe("rgb(55, 102, 167)");
  });

  it("renders intermediate magnitudes between neutral and the extreme color", () => {
    const color = continuousFieldColor(2, 8);
    expect(color).not.toBe("rgb(184, 181, 173)");
    expect(color).not.toBe("rgb(200, 75, 64)");
  });
});
