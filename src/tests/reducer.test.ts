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
});
