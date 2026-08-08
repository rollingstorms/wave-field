import { describe, expect, it } from "vitest";
import { DEFAULT_DEFINITIONS, validateDefinition } from "../field/componentDefinitions";
import { evaluateField, evaluatePieceContribution, evaluateTypeFields } from "../field/evaluateField";
import { evaluateBasis, evaluateComponentBasis } from "../field/kernels";
import { DEFAULT_HOME_ENERGY, FIELD_EPSILON } from "../game/constants";
import { createInitialState } from "../game/initialState";
import type { BasisDefinition, Coefficient, FormulaPreset, GameState, PieceType } from "../game/types";

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

function defaultValues(pieceType: PieceType): Coefficient[] {
  switch (pieceType) {
    case "pawn":
      return [1];
    case "rook":
      return [1, 1];
    case "spy":
      return [1, 0];
    case "king":
      return [1, 1];
  }
}

describe("field engine", () => {
  it("zero pawn coefficient contributes zero away from its home square", () => {
    const state = tuned("pawn", [0]);
    const piece = state.pieces[0];
    for (let y = 0; y < 7; y += 1) {
      for (let x = 0; x < 7; x += 1) {
        if (x !== piece.position.x || y !== piece.position.y) {
          expect(evaluatePieceContribution(piece, { x, y }, state)).toBe(0);
        }
      }
    }
  });

  it("pawn defaults to stronger friendly than hostile ring-1 pressure", () => {
    const state = tuned("pawn", [1]);
    const piece = state.pieces[0];
    expect(evaluatePieceContribution(piece, { x: 4, y: 3 }, state)).toBeCloseTo(-0.5);
    expect(evaluatePieceContribution(piece, { x: 4, y: 4 }, state)).toBeCloseTo(2);
  });

  it("decay halves each ring for equal signs", () => {
    const basin = DEFAULT_DEFINITIONS.king[0];
    expect(Math.abs(evaluateBasis(basin, { x: 1, y: 0 }))).toBeCloseTo(Math.abs(evaluateBasis(basin, { x: 0, y: 0 })) / 2);
    expect(Math.abs(evaluateBasis(basin, { x: 2, y: 0 }))).toBeCloseTo(Math.abs(evaluateBasis(basin, { x: 1, y: 0 })) / 2);
  });

  it("grid definitions use integer cells before normal ring decay", () => {
    const gridValues = Array.from({ length: 7 }, () => Array.from({ length: 7 }, () => 0));
    gridValues[3][4] = 2;
    gridValues[3][6] = -8;
    const definition: BasisDefinition = { kind: "grid", name: "Counter-decay", gridValues, decayBase: 2, originScale: 1 };

    expect(evaluateBasis(definition, { x: 1, y: 0 })).toBeCloseTo(1);
    expect(evaluateBasis(definition, { x: 3, y: 0 })).toBeCloseTo(-1);
    expect(evaluateBasis(definition, { x: 4, y: 0 })).toBe(0);
    expect(validateDefinition(definition)).toBe(true);
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

  it("rook +- scales after summing the component basis values", () => {
    const state = tuned("rook", [1, -1]);
    const piece = state.pieces[0];
    const square = { x: 5, y: 3 };
    const delta = { x: 2, y: 0 };
    const c1 = evaluateBasis(DEFAULT_DEFINITIONS.rook[0], delta);
    const c2 = -evaluateBasis(DEFAULT_DEFINITIONS.rook[1], delta);
    const rawValue = c1 + c2;
    const scale = rawValue >= 0 ? state.waveScales.rook.friendly : state.waveScales.rook.hostile;
    const expected = 2 * rawValue * scale;
    expect(evaluatePieceContribution(piece, square, state)).toBeCloseTo(expected);
  });

  it("spy +00 uses the spy friendly scale on the near remote scout basis", () => {
    const spy = tuned("spy", [1, 0, 0]);
    expect(evaluatePieceContribution(spy.pieces[0], { x: 2, y: 2 }, spy)).toBeCloseTo(3);
  });

  it("piece home squares use the configured home energy", () => {
    (["pawn", "rook", "spy", "king"] as PieceType[]).forEach((pieceType) => {
      const state = tuned(pieceType, defaultValues(pieceType));
      const piece = state.pieces[0];
      const typeFields = evaluateTypeFields(state);

      expect(evaluatePieceContribution(piece, piece.position, state)).toBe(DEFAULT_HOME_ENERGY[pieceType]);
      expect(typeFields[pieceType][piece.position.y][piece.position.x]).toBe(DEFAULT_HOME_ENERGY[pieceType]);
    });
  });

  it("home square contribution ignores tuned component values", () => {
    const spy = tuned("spy", [0, 0, 0]);
    const rook = tuned("rook", [-1, -1]);
    expect(evaluatePieceContribution(spy.pieces[0], spy.pieces[0].position, spy)).toBe(0.5);
    expect(evaluatePieceContribution(rook.pieces[0], rook.pieces[0].position, rook)).toBe(0);
  });

  it("own-square contribution is zero while adjacent contribution still works", () => {
    const king = tuned("king", [0, 1, 1]);
    king.pieces[0].type = "king";
    const pawn = tuned("pawn", [1]);
    const typeFields = evaluateTypeFields(king);

    expect(evaluatePieceContribution(king.pieces[0], { x: 3, y: 3 }, king)).toBe(DEFAULT_HOME_ENERGY.king);
    expect(typeFields.king[3][3]).toBe(DEFAULT_HOME_ENERGY.king);
    expect(evaluatePieceContribution(king.pieces[0], { x: 4, y: 3 }, king)).not.toBe(0);
    expect(evaluatePieceContribution(pawn.pieces[0], { x: 3, y: 3 }, pawn)).toBe(0);
    expect(evaluatePieceContribution(pawn.pieces[0], { x: 4, y: 3 }, pawn)).not.toBe(0);
  });

  it("component basis omits ring zero for every piece type", () => {
    const blockChecker = DEFAULT_DEFINITIONS.king[2];
    expect(evaluateBasis(blockChecker, { x: 0, y: 0 })).not.toBe(0);
    (["pawn", "rook", "spy", "king"] as PieceType[]).forEach((pieceType) => {
      expect(evaluateComponentBasis(pieceType, blockChecker, { x: 0, y: 0 })).toBe(0);
    });
    expect(evaluateComponentBasis("rook", blockChecker, { x: 1, y: 0 })).not.toBe(0);
  });

  it("diamond core is distinct from the checkerboard preset", () => {
    const checkerboard = DEFAULT_DEFINITIONS.pawn[0];
    const diamondCore = DEFAULT_DEFINITIONS.spy[1];

    expect(evaluateBasis(checkerboard, { x: 1, y: 0 })).toBeLessThan(0);
    expect(evaluateBasis(diamondCore, { x: 1, y: 0 })).toBeGreaterThan(0);
    expect(evaluateBasis(diamondCore, { x: 3, y: 0 })).toBeLessThan(0);
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
