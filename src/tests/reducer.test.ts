import { describe, expect, it } from "vitest";
import { evaluateField } from "../field/evaluateField";
import { gameReducer } from "../game/reducer";
import { createInitialState } from "../game/initialState";
import { getLegalMoves } from "../game/movement";
import { beginTurn } from "../game/rules";

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
    expect(kingFlipped.components.blue.king).toEqual([0, -1, 1]);
    expect(kingFlipped.currentPlayer).toBe("blue");
  });

  it("allows temporary self-trapping tuning before the turn-ending move", () => {
    const state = createInitialState();
    const tuned = gameReducer(state, { type: "tune", pieceType: "king", componentIndex: 1, value: -1 });

    expect(tuned.components.blue.king).toEqual([0, -1, 1]);
    expect(tuned.message).toContain("move a piece to end the turn");
  });

  it("rejects activating the king's neutral C1 while already at full strength", () => {
    const state = createInitialState();
    const tuned = gameReducer(state, { type: "tune", pieceType: "king", componentIndex: 0, value: -1 });

    expect(tuned.components.blue.king).toEqual([0, 1, 1]);
    expect(tuned.message).toContain("up to 2 active components");
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
    const expected = beginTurn(createInitialState(edited.defaultComponents, edited.definitions));

    expect(restarted.definitions.pawn[0].name).toBe("Flat");
    expect(restarted.status).toBe(expected.status);
    expect(restarted.message).toBe(expected.message);
    expect(restarted.pieces.map((piece) => piece.unstable)).toEqual(expected.pieces.map((piece) => piece.unstable));
  });

  it("applies edited default controls to both players on restart", () => {
    const state = createInitialState();
    const edited = gameReducer(state, { type: "update-default-component", pieceType: "king", componentIndex: 2, value: 0 });

    expect(edited.components.blue.king).toEqual([0, 1, 1]);
    expect(edited.defaultComponents.king).toEqual([0, 1, 0]);

    const restarted = gameReducer(edited, { type: "restart", keepDefinitions: true });
    expect(restarted.components.blue.king).toEqual([0, 1, 0]);
    expect(restarted.components.red.king).toEqual([0, 1, 0]);
  });

  it("keeps edited defaults when undoing a game action", () => {
    const state = createInitialState();
    const tuned = gameReducer(state, { type: "tune", pieceType: "pawn", componentIndex: 0, value: -1 });
    const edited = gameReducer(tuned, { type: "update-default-component", pieceType: "king", componentIndex: 2, value: 0 });
    const undone = gameReducer(edited, { type: "undo" });

    expect(undone.components.blue.pawn).toEqual([1]);
    expect(undone.defaultComponents.king).toEqual([0, 1, 0]);
  });

  it("updates wave scales with undo history and preserves them on restart", () => {
    const state = createInitialState();
    const scaled = gameReducer(state, { type: "update-wave-scale", pieceType: "rook", scale: "hostile", value: 1.5 });

    expect(scaled.waveScales.rook.hostile).toBe(1.5);
    expect(scaled.history).toHaveLength(1);

    const restarted = gameReducer(scaled, { type: "restart", keepDefinitions: true });
    expect(restarted.waveScales.rook.hostile).toBe(1.5);

    const undone = gameReducer(scaled, { type: "undo" });
    expect(undone.waveScales.rook.hostile).toBe(1);
  });

  it("rejects default controls beyond a piece's strength", () => {
    const state = createInitialState();
    const first = gameReducer(state, { type: "update-default-component", pieceType: "spy", componentIndex: 1, value: -1 });

    expect(first.defaultComponents.spy).toEqual([1, 0, 0]);
    expect(first.message).toContain("exceed its active limit");
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
