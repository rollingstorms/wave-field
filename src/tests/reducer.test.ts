import { describe, expect, it } from "vitest";
import { evaluateField } from "../field/evaluateField";
import { gameReducer } from "../game/reducer";
import { createInitialState } from "../game/initialState";
import { getLegalMoves } from "../game/movement";

describe("reducer", () => {
  it("allows multiple tuning changes without ending the turn", () => {
    const state = createInitialState();
    const tuned = gameReducer(state, { type: "tune", pieceType: "pawn", componentIndex: 0, value: 0 });
    const tunedAgain = gameReducer(tuned, { type: "tune", pieceType: "spy", componentIndex: 0, value: 0 });

    expect(tunedAgain.currentPlayer).toBe("blue");
    expect(tunedAgain.components.blue.pawn[0]).toBe(0);
    expect(tunedAgain.components.blue.spy[0]).toBe(0);
    expect(tunedAgain.history).toHaveLength(2);
  });

  it("ends the turn only after a piece moves", () => {
    const state = createInitialState();
    const tuned = gameReducer(state, { type: "tune", pieceType: "pawn", componentIndex: 0, value: 0 });
    const destination = getLegalMoves("blue-pawn-1", tuned, evaluateField(tuned))[0];
    expect(destination).toBeDefined();

    const moved = gameReducer(tuned, { type: "move", pieceId: "blue-pawn-1", destination });
    expect(moved.currentPlayer).toBe("red");
  });

  it("rejects tuning beyond a piece's strength", () => {
    const state = createInitialState();
    const first = gameReducer(state, { type: "tune", pieceType: "spy", componentIndex: 1, value: 1 });

    expect(first.components.blue.spy).toEqual([1, 0, 0]);
    expect(first.message).toContain("up to 1 active component");
  });

  it("allows active pawn and king components to flip sign", () => {
    const state = createInitialState();
    const pawnFlipped = gameReducer(state, { type: "tune", pieceType: "pawn", componentIndex: 0, value: -1 });
    const kingFlipped = gameReducer(pawnFlipped, { type: "tune", pieceType: "king", componentIndex: 1, value: -1 });

    expect(pawnFlipped.components.blue.pawn).toEqual([-1]);
    expect(kingFlipped.components.blue.king).toEqual([0, -1, 0]);
    expect(kingFlipped.currentPlayer).toBe("blue");
  });

  it("allows temporary self-trapping tuning before the turn-ending move", () => {
    const state = createInitialState();
    const tuned = gameReducer(state, { type: "tune", pieceType: "king", componentIndex: 1, value: -1 });

    expect(tuned.components.blue.king).toEqual([0, -1, 0]);
    expect(tuned.message).toContain("move a piece to end the turn");
  });

  it("rejects tuning the king's locked C1 component", () => {
    const state = createInitialState();
    const tuned = gameReducer(state, { type: "tune", pieceType: "king", componentIndex: 0, value: 1 });

    expect(tuned.components.blue.king).toEqual([0, 1, 0]);
    expect(tuned.message).toContain("fixed at Neutral");
  });

  it("undo restores the previous in-turn tuning state", () => {
    const state = createInitialState();
    const tuned = gameReducer(state, { type: "tune", pieceType: "pawn", componentIndex: 0, value: 0 });
    const undone = gameReducer(tuned, { type: "undo" });

    expect(undone.currentPlayer).toBe("blue");
    expect(undone.components.blue.pawn[0]).toBe(1);
  });

  it("restart can preserve edited definitions", () => {
    const state = createInitialState();
    const edited = gameReducer(state, {
      type: "update-definition",
      pieceType: "pawn",
      componentIndex: 0,
      definition: { kind: "preset", name: "Flat", preset: "constant-basin", decayBase: 2, originScale: 1 },
    });
    const restarted = gameReducer(edited, { type: "restart", keepDefinitions: true });
    expect(restarted.definitions.pawn[0].name).toBe("Flat");
  });

  it("rejects wave definitions that break base-2 decay", () => {
    const state = createInitialState();
    const edited = gameReducer(state, {
      type: "update-definition",
      pieceType: "pawn",
      componentIndex: 0,
      definition: { kind: "preset", name: "Thirds", preset: "constant-basin", decayBase: 3, originScale: 1 },
    });

    expect(edited.definitions.pawn[0]).toEqual(state.definitions.pawn[0]);
    expect(edited.message).toContain("base-2 decay");
  });
});
