import { describe, expect, it } from "vitest";
import { DEFAULT_DEFINITIONS, validateDefinition } from "../field/componentDefinitions";
import { evaluateField, evaluatePieceContribution, evaluateTypeFields } from "../field/evaluateField";
import { evaluateBasis } from "../field/kernels";
import { FIELD_EPSILON } from "../game/constants";
import { createInitialState } from "../game/initialState";
import type { Coefficient, FormulaPreset, GameState, PieceType } from "../game/types";

const allPresets: FormulaPreset[] = [
  "checkerboard",
  "diagonal-stripes",
  "horizontal-versus-vertical",
  "quadrants",
  "constant-basin",
  "skipped-rings",
  "compass-rose",
  "axis-favor",
  "diagonal-favor",
  "wide-bullseye",
  "pulse-gap",
  "block-checker",
  "diamond-core",
  "astigmatism",
  "local-flip",
  "adjacent-opinion",
  "sink",
  "deep-sink",
  "far-crown",
  "slow-governance",
  "dipole-x",
  "dipole-y",
];

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

  it("every default basis value is zero or a signed power of two", () => {
    const definitions = Object.values(DEFAULT_DEFINITIONS).flat();
    for (const definition of definitions) {
      for (let y = -6; y <= 6; y += 1) {
        for (let x = -6; x <= 6; x += 1) {
          const magnitude = Math.abs(evaluateBasis(definition, { x, y }));
          expect(magnitude === 0 || Number.isInteger(Math.log2(magnitude))).toBe(true);
        }
      }
    }
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

  it("diagonal stripes are distinct from the checkerboard preset", () => {
    const checkerboard = DEFAULT_DEFINITIONS.pawn[0];
    const diagonalStripes = DEFAULT_DEFINITIONS.spy[1];

    expect(evaluateBasis(checkerboard, { x: 1, y: 0 })).toBeLessThan(0);
    expect(evaluateBasis(diagonalStripes, { x: 1, y: 0 })).toBeGreaterThan(0);
    expect(evaluateBasis(diagonalStripes, { x: 2, y: 0 })).toBeLessThan(0);
  });

  it("all formula presets evaluate and validate", () => {
    for (const preset of allPresets) {
      const definition = { kind: "preset" as const, name: preset, preset, decayBase: 2, originScale: 1 };
      expect(validateDefinition(definition)).toBe(true);
      expect([-1, 0, 1]).toContain(Math.sign(evaluateBasis(definition, { x: 2, y: 1 })));
    }
  });

  it("removing a piece removes exactly its contribution", () => {
    const state = createInitialState();
    const removed = state.pieces[0];
    const full = evaluateField(state);
    const without = evaluateField({ ...state, pieces: state.pieces.slice(1) });
    const contribution = -evaluatePieceContribution(removed, { x: 3, y: 3 }, state);
    expect(full[3][3] - without[3][3]).toBeCloseTo(contribution);
  });

  it("piece-type fields sum back to the complete field", () => {
    const state = createInitialState();
    const field = evaluateField(state);
    const typeFields = evaluateTypeFields(state);
    for (let y = 0; y < 7; y += 1) {
      for (let x = 0; x < 7; x += 1) {
        expect(
          typeFields.pawn[y][x]
          + typeFields.rook[y][x]
          + typeFields.spy[y][x]
          + typeFields.king[y][x],
        ).toBeCloseTo(field[y][x]);
      }
    }
  });

  it("every generated field total is a dyadic rational", () => {
    const state = createInitialState();
    const values = [
      ...evaluateField(state).flat(),
      ...Object.values(evaluateTypeFields(state)).flatMap((field) => field.flat()),
    ];
    for (const value of values) {
      const becomesInteger = Array.from(
        { length: 13 },
        (_, exponent) => value * Math.pow(2, exponent),
      ).some(Number.isInteger);
      expect(becomesInteger).toBe(true);
    }
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
