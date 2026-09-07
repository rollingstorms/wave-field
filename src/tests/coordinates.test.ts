import { describe, expect, it } from "vitest";
import { boardCoordinate } from "../game/coordinates";

describe("board coordinates", () => {
  it("uses square notation for integer board centers", () => {
    expect(boardCoordinate({ x: 2, y: 1 })).toBe("C6");
  });

  it("uses trimmed precise notation for fractional points", () => {
    expect(boardCoordinate({ x: 5, y: 3 + 1 / 9 })).toBe("(5, 3.111)");
  });
});
