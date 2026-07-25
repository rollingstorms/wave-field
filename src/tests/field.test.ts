import { describe, expect, it } from "vitest";
import { DEFAULT_DEFINITIONS } from "../field/componentDefinitions";
import { evaluateField, evaluatePieceContribution } from "../field/evaluateField";
import { evaluateBasis } from "../field/kernels";
import { FIELD_EPSILON } from "../game/constants";
import { createInitialState } from "../game/initialState";
import type { Coefficient, GameState, PieceType } from "../game/types";

function tuned(pieceType: PieceType, values: Coefficient[]): GameState {
  const state = createInitialState();
  state.pieces = [{ id: "red-pawn-1", owner: "red", type: pieceType, position: { x: 3, y: 3 }, unstable: false }];
  state.components.red[pieceType] = values as never;
  return state;
}

describe("field engine", () => {
  it("zero pawn coefficient contributes zero everywhere", () => {
    const state = tuned("pawn", [0]);
    const piece = state.pieces[0];
    for (let y = 0; y < 7; y += 1) {
      for (let x = 0; x < 7; x += 1) {
        expect(evaluatePieceContribution(piece, { x, y }, state)).toBe(0);
      }
    }
  });

  it("negative pawn coefficient is the exact inverse of positive", () => {
    const positive = tuned("pawn", [1]);
    const negative = tuned("pawn", [-1]);
    const p = positive.pieces[0];
    const n = negative.pieces[0];
    expect(evaluatePieceContribution(p, { x: 4, y: 4 }, positive)).toBe(-evaluatePieceContribution(n, { x: 4, y: 4 }, negative));
  });

  it("decay halves each ring for equal signs", () => {
    const basin = DEFAULT_DEFINITIONS.king[0];
    expect(Math.abs(evaluateBasis(basin, { x: 1, y: 0 }))).toBeCloseTo(Math.abs(evaluateBasis(basin, { x: 0, y: 0 })) / 2);
    expect(Math.abs(evaluateBasis(basin, { x: 2, y: 0 }))).toBeCloseTo(Math.abs(evaluateBasis(basin, { x: 1, y: 0 })) / 2);
  });

  it("rook +- equals strength times basis difference", () => {
    const state = tuned("rook", [1, -1]);
    const piece = state.pieces[0];
    const square = { x: 5, y: 3 };
    const delta = { x: 2, y: 0 };
    const expected = 2 * (evaluateBasis(DEFAULT_DEFINITIONS.rook[0], delta) - evaluateBasis(DEFAULT_DEFINITIONS.rook[1], delta));
    expect(evaluatePieceContribution(piece, square, state)).toBeCloseTo(expected);
  });

  it("spy +00 matches the pawn geometric basis before strength differences", () => {
    const spy = tuned("spy", [1, 0, 0]);
    const pawn = tuned("pawn", [1]);
    expect(evaluatePieceContribution(spy.pieces[0], { x: 2, y: 2 }, spy)).toBeCloseTo(evaluatePieceContribution(pawn.pieces[0], { x: 2, y: 2 }, pawn));
  });

  it("removing a piece removes exactly its contribution", () => {
    const state = createInitialState();
    const removed = state.pieces[0];
    const full = evaluateField(state);
    const without = evaluateField({ ...state, pieces: state.pieces.slice(1) });
    const contribution = -evaluatePieceContribution(removed, { x: 3, y: 3 }, state);
    expect(full[3][3] - without[3][3]).toBeCloseTo(contribution);
  });

  it("projection epsilon stays neutral around zero", async () => {
    const { projectFieldValue } = await import("../field/projection");
    expect(projectFieldValue(1)).toBe("red");
    expect(projectFieldValue(FIELD_EPSILON / 2)).toBe("neutral");
    expect(projectFieldValue(0)).toBe("neutral");
    expect(projectFieldValue(-FIELD_EPSILON / 2)).toBe("neutral");
    expect(projectFieldValue(-1)).toBe("blue");
  });
});
