import { describe, expect, it } from "vitest";
import { evaluateField } from "../field/evaluateField";
import { gameReducer } from "../game/reducer";
import { createInitialState } from "../game/initialState";
import { getLegalMoves } from "../game/movement";
import { beginTurn } from "../game/rules";

describe("reducer", () => {
  it("allows multiple tuning changes without ending the turn", () => {
    const state = createInitialState();
    const tuned = gameReducer(state, { type: "tune", pieceType: "pawn", componentIndex: 0, value: -1 });
    const tunedAgain = gameReducer(tuned, { type: "tune", pieceType: "spy", componentIndex: 1, value: 1 });

    expect(tunedAgain.currentPlayer).toBe("blue");
    expect(tunedAgain.components.blue.pawn[0]).toBe(-1);
    expect(tunedAgain.components.blue.spy).toEqual([0, 1, 0]);
    expect(tunedAgain.history).toHaveLength(2);
  });

  it("ends the turn only after a piece moves", () => {
    const state = createInitialState();
    const tuned = gameReducer(state, { type: "tune", pieceType: "pawn", componentIndex: 0, value: -1 });
    const destination = getLegalMoves("blue-pawn-1", tuned, evaluateField(tuned))[0];
    expect(destination).toBeDefined();

    const moved = gameReducer(tuned, { type: "move", pieceId: "blue-pawn-1", destination });
    expect(moved.currentPlayer).toBe("red");
  });

  it("keeps only the last activated spy component", () => {
    const state = createInitialState();
    const first = gameReducer(state, { type: "tune", pieceType: "spy", componentIndex: 1, value: 1 });

    expect(first.components.blue.spy).toEqual([0, 1, 0]);
    expect(first.activationOrders.blue.spy).toEqual([1]);
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

  it("evicts the oldest active king component at full strength", () => {
    const state = createInitialState();
    const tuned = gameReducer(state, { type: "tune", pieceType: "king", componentIndex: 0, value: -1 });

    expect(tuned.components.blue.king).toEqual([-1, 0, 1]);
    expect(tuned.activationOrders.blue.king).toEqual([2, 0]);
  });

  it("rejects clearing an active component", () => {
    const state = createInitialState();
    const tuned = gameReducer(state, { type: "tune", pieceType: "pawn", componentIndex: 0, value: 0 });

    expect(tuned.components.blue.pawn).toEqual([1]);
    expect(tuned.activationOrders.blue.pawn).toEqual([0]);
    expect(tuned.message).toContain("full strength");
  });

  it("a sign flip makes the component most recently active", () => {
    const state = createInitialState();
    const flipped = gameReducer(state, { type: "tune", pieceType: "king", componentIndex: 1, value: -1 });
    const activated = gameReducer(flipped, { type: "tune", pieceType: "king", componentIndex: 0, value: -1 });

    expect(flipped.activationOrders.blue.king).toEqual([2, 1]);
    expect(activated.components.blue.king).toEqual([-1, -1, 0]);
    expect(activated.activationOrders.blue.king).toEqual([1, 0]);
  });

  it("undo restores the previous in-turn tuning state", () => {
    const state = createInitialState();
    const tuned = gameReducer(state, { type: "tune", pieceType: "pawn", componentIndex: 0, value: -1 });
    const undone = gameReducer(tuned, { type: "undo" });

    expect(undone.currentPlayer).toBe("blue");
    expect(undone.components.blue.pawn[0]).toBe(1);
    expect(undone.activationOrders.blue.pawn).toEqual([0]);
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
    const edited = gameReducer(state, { type: "update-default-component", pieceType: "king", componentIndex: 0, value: -1 });

    expect(edited.components.blue.king).toEqual([0, 1, 1]);
    expect(edited.defaultComponents.king).toEqual([-1, 0, 1]);

    const restarted = gameReducer(edited, { type: "restart", keepDefinitions: true });
    expect(restarted.components.blue.king).toEqual([-1, 0, 1]);
    expect(restarted.components.red.king).toEqual([-1, 0, 1]);
  });

  it("keeps edited defaults when undoing a game action", () => {
    const state = createInitialState();
    const tuned = gameReducer(state, { type: "tune", pieceType: "pawn", componentIndex: 0, value: -1 });
    const edited = gameReducer(tuned, { type: "update-default-component", pieceType: "king", componentIndex: 0, value: -1 });
    const undone = gameReducer(edited, { type: "undo" });

    expect(undone.components.blue.pawn).toEqual([1]);
    expect(undone.defaultComponents.king).toEqual([-1, 0, 1]);
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
    expect(undone.waveScales.rook.friendly).toBe(3);
  });

  it("updates home energy with undo history and preserves it on restart", () => {
    const state = createInitialState();
    const homed = gameReducer(state, { type: "update-home-energy", pieceType: "spy", value: 0.75 });

    expect(homed.homeEnergy.spy).toBe(0.75);
    expect(homed.history).toHaveLength(1);

    const restarted = gameReducer(homed, { type: "restart", keepDefinitions: true });
    expect(restarted.homeEnergy.spy).toBe(0.75);

    const reset = gameReducer(homed, { type: "reset-home-energy" });
    expect(reset.homeEnergy.spy).toBe(0.5);

    const undone = gameReducer(homed, { type: "undo" });
    expect(undone.homeEnergy.spy).toBe(0.5);
  });

  it("replaces an existing default control when a type is at full strength", () => {
    const state = createInitialState();
    const first = gameReducer(state, { type: "update-default-component", pieceType: "spy", componentIndex: 1, value: -1 });

    expect(first.defaultComponents.spy).toEqual([0, -1, 0]);
    expect(first.message).toContain("updated");
  });

  it("rejects zero default controls", () => {
    const state = createInitialState();
    const first = gameReducer(state, { type: "update-default-component", pieceType: "pawn", componentIndex: 0, value: 0 });

    expect(first.defaultComponents.pawn).toEqual([1]);
    expect(first.message).toContain("full strength");
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
