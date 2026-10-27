import { describe, expect, it } from "vitest";
import { evaluateField, evaluatePieceContribution, evaluateTeamFields } from "../field/evaluateField";
import { createInitialState } from "../game/initialState";
import { getLegalMoves } from "../game/movement";
import { gameReducer } from "../game/reducer";

function sparseState() {
  const state = createInitialState();
  state.variant = "sparse";
  return state;
}

describe("sparse variant", () => {
  it("thresholds existing waves per piece into small integers and anchors the home square", () => {
    const state = sparseState();
    const piece = state.pieces.find((candidate) => candidate.id === "red-pawn-1")!;
    piece.position = { x: 3, y: 3 };
    state.pieces = [piece];

    expect(evaluatePieceContribution(piece, { x: 3, y: 3 }, state)).toBe(1);
    expect(evaluatePieceContribution(piece, { x: 4, y: 3 }, state)).toBe(0);
    expect(evaluatePieceContribution(piece, { x: 4, y: 4 }, state)).toBe(2);
    expect(evaluatePieceContribution(piece, { x: 5, y: 3 }, state)).toBe(1);
    expect(evaluatePieceContribution(piece, { x: 6, y: 3 }, state)).toBe(0);
    expect(evaluatePieceContribution(piece, { x: 2, y: 3 }, state)).toBe(0);
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
    const moved = structuredClone(state);
    moved.pieces.find((candidate) => candidate.id === piece.id)!.position = { x: 0, y: 6 };
    expect(evaluatePieceContribution(piece, origin, state)).toBe(1);
    expect(evaluatePieceContribution(moved.pieces.find((candidate) => candidate.id === piece.id)!, origin, moved)).toBe(0);
  });

  it("has a playable opening and preserves the variant on restart", () => {
    const state = sparseState();
    const openingMoves = state.pieces
      .filter((piece) => piece.owner === "blue")
      .flatMap((piece) => getLegalMoves(piece.id, state, evaluateField(state)));
    expect(openingMoves.length).toBeGreaterThan(0);
    expect(gameReducer(state, { type: "restart", keepDefinitions: true }).variant).toBe("sparse");
  });
});
