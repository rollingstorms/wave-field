import { describe, expect, it } from "vitest";
import { evaluateField, evaluatePieceContribution, evaluateTeamFields } from "../field/evaluateField";
import { createInitialState, fromSnapshot, snapshot } from "../game/initialState";
import { playHeuristicTurn } from "../game/ai";
import { getLegalMoves } from "../game/movement";
import { gameReducer } from "../game/reducer";
import { sparsePatternCells } from "../field/sparsePattern";
import type { Coefficient, PieceType } from "../game/types";

function sparseState() {
  return createInitialState();
}

describe("sparse variant", () => {
  it("is the engine default and preserves the ruleset through undo and restart", () => {
    const state = sparseState();
    expect(state.variant).toBe("sparse");
    expect(state.homeEnergy).toEqual({ pawn: 0, rook: 0, spy: 1, king: 0 });
    const tuned = gameReducer(state, { type: "tune", pieceType: "rook", componentIndex: 0, value: -1 });
    expect(gameReducer(tuned, { type: "undo" }).variant).toBe("sparse");
    expect(gameReducer(tuned, { type: "restart" }).variant).toBe("sparse");
    const legacy = snapshot(state);
    delete (legacy as Partial<typeof legacy>).variant;
    expect(fromSnapshot(legacy).variant).toBe("classic");
  });

  it("gives every piece a discrete wide and focused footprint", () => {
    const cases = [
      { type: "pawn" as const, wide: [1], focused: [-1], wideCount: 8, focusedCount: 4, focusedStrength: 2, home: 0 },
      { type: "spy" as const, wide: [1], focused: [-1], wideCount: 12, focusedCount: 4, focusedStrength: 3, home: 1 },
      { type: "rook" as const, wide: [1, 1], focused: [-1, -1], wideCount: 16, focusedCount: 4, focusedStrength: 4, home: 0 },
      { type: "king" as const, wide: [1, 1], focused: [-1, -1], wideCount: 24, focusedCount: 4, focusedStrength: 6, home: 0 },
    ];
    for (const entry of cases) {
      const state = sparseState();
      const piece = state.pieces.find((candidate) => candidate.owner === "red" && candidate.type === entry.type)!;
      piece.position = { x: 3, y: 3 };
      state.pieces = [piece];
      const count = (coefficients: number[]) => {
        state.components.red[entry.type] = coefficients as (1 | -1)[];
        const values = Array.from({ length: 7 }, (_, y) => Array.from({ length: 7 }, (_, x) => evaluatePieceContribution(piece, { x, y }, state)));
        expect(values[3][3]).toBe(entry.home);
        expect(values.flat().every((value) => Number.isInteger(value) && value >= 0 && value <= 6)).toBe(true);
        return values.flat().filter((value, index) => index !== 24 && value > 0);
      };
      const wide = count(entry.wide);
      const focused = count(entry.focused);
      expect(wide).toHaveLength(entry.wideCount);
      expect(wide.every((value) => value === 1)).toBe(true);
      expect(focused).toHaveLength(entry.focusedCount);
      expect(focused.every((value) => value === entry.focusedStrength)).toBe(true);
    }
  });

  it("maps the mixed two-control modes to distinct, narrower shapes", () => {
    for (const type of ["rook", "king"] as const) {
      const state = sparseState();
      const piece = state.pieces.find((candidate) => candidate.owner === "red" && candidate.type === type)!;
      piece.position = { x: 3, y: 3 };
      state.pieces = [piece];
      const footprint = (profile: (1 | -1)[]) => {
        state.components.red[type] = profile;
        const values = Array.from({ length: 7 }, (_, y) => Array.from({ length: 7 }, (_, x) => evaluatePieceContribution(piece, { x, y }, state)));
        return { count: values.flat().filter((value, index) => index !== 24 && value > 0).length, values };
      };
      const wide = footprint([1, 1]);
      const mixedCross = footprint([1, -1]);
      const mixedDiagonal = footprint([-1, 1]);
      const tight = footprint([-1, -1]);
      expect(wide.count).toBeGreaterThan(mixedCross.count);
      expect(mixedCross.count).toBeGreaterThanOrEqual(mixedDiagonal.count);
      expect(mixedDiagonal.count).toBeGreaterThan(tight.count);
      expect(mixedCross.values[3][4]).toBe(2);
      expect(mixedDiagonal.values[4][4]).toBe(type === "king" ? 3 : 2);
      expect(tight.values[3][4]).toBe(type === "king" ? 6 : 4);
      expect(tight.values[4][4]).toBe(0);
      expect(tight.values[3][5]).toBe(0);
    }
  });

  it("preserves total support across every full tuning profile", () => {
    const cases: Array<{ type: PieceType; profiles: Coefficient[][]; total: number }> = [
      { type: "pawn", profiles: [[1], [-1]], total: 8 },
      { type: "spy", profiles: [[1], [-1]], total: 13 },
      { type: "rook", profiles: [[1, 1], [1, -1], [-1, 1], [-1, -1]], total: 16 },
      { type: "king", profiles: [[1, 1], [1, -1], [-1, 1], [-1, -1]], total: 24 },
    ];
    for (const { type, profiles, total } of cases) {
      for (const profile of profiles) {
        const cells = sparsePatternCells(type, profile);
        expect(cells.reduce((sum, cell) => sum + cell.support, 0), `${type} ${profile}`).toBe(total);
        const home = cells.find((cell) => cell.x === 0 && cell.y === 0)?.support ?? 0;
        expect(home).toBe(type === "spy" ? 1 : 0);
      }
    }
  });

  it("shows team totals that exactly explain territory and removes a moved pattern from its old origin", () => {
    const state = sparseState();
    const before = evaluateTeamFields(state);
    const field = evaluateField(state);
    for (let y = 0; y < field.length; y += 1) {
      for (let x = 0; x < field[y].length; x += 1) {
        expect(field[y][x]).toBe(before.red[y][x] - before.blue[y][x]);
        expect(Number.isInteger(field[y][x])).toBe(true);
        expect(before.red[y][x]).toBeGreaterThanOrEqual(0);
        expect(before.blue[y][x]).toBeGreaterThanOrEqual(0);
      }
    }

    const piece = state.pieces.find((candidate) => candidate.id === "blue-pawn-1")!;
    const origin = { ...piece.position };
    const oldPatternSquare = { x: origin.x + 1, y: origin.y };
    const moved = structuredClone(state);
    moved.pieces.find((candidate) => candidate.id === piece.id)!.position = { x: 0, y: 6 };
    expect(evaluatePieceContribution(piece, origin, state)).toBe(0);
    expect(evaluatePieceContribution(piece, oldPatternSquare, state)).toBe(1);
    expect(evaluatePieceContribution(moved.pieces.find((candidate) => candidate.id === piece.id)!, oldPatternSquare, moved)).toBe(0);
  });

  it("has a playable opening and preserves the variant on restart", () => {
    const state = sparseState();
    const openingMoves = state.pieces
      .filter((piece) => piece.owner === "blue")
      .flatMap((piece) => getLegalMoves(piece.id, state, evaluateField(state)));
    expect(openingMoves.length).toBeGreaterThan(0);
    expect(gameReducer(state, { type: "restart", keepDefinitions: true }).variant).toBe("sparse");
  });

  it("lets the fallback AI complete a turn under the default rules", () => {
    const state = sparseState();
    const result = playHeuristicTurn(state, "blue", { seed: 17, timeBudgetMs: 1_000 });
    expect(result.variant).toBe("sparse");
    expect(result.currentPlayer).toBe("red");
    expect(result.turnNumber).toBe(state.turnNumber);
    expect(result.status).toBe("playing");
    expect(result.pieces.some((piece) => {
      const before = state.pieces.find((candidate) => candidate.id === piece.id)!;
      return before.position.x !== piece.position.x || before.position.y !== piece.position.y;
    })).toBe(true);
  });
});
