import { describe, expect, it } from "vitest";
import { isKingUnprotected, getUnstablePieces, removeUnrescuedPieces } from "../game/victory";
import { createInitialState } from "../game/initialState";
import { applyClosestPlayableHint, applyHintSearch, applyMove, beginTurn, findClosestPlayableConfiguration, getPlayableMoves, resignInCheck } from "../game/rules";
import type { GameState } from "../game/types";
import { evaluateField } from "../field/evaluateField";

const field = (value = 0) => Array.from({ length: 7 }, () => Array.from({ length: 7 }, () => value));

function kingState(): GameState {
  const state = createInitialState();
  state.pieces = [{ id: "red-king", owner: "red", type: "king", position: { x: 3, y: 3 }, unstable: false }];
  return state;
}

describe("stability and victory", () => {
  it("spies become unstable on hostile territory", () => {
    const state = createInitialState();
    state.pieces = [{ id: "red-spy", owner: "red", type: "spy", position: { x: 3, y: 3 }, unstable: false }];
    expect(getUnstablePieces("red", state, field(-1)).map((piece) => piece.id)).toEqual(["red-spy"]);
  });

  it("an unrescued spy still on hostile territory is lost", () => {
    const state = createInitialState();
    state.pieces = [
      { id: "red-spy", owner: "red", type: "spy", position: { x: 1, y: 1 }, unstable: true },
      { id: "red-king", owner: "red", type: "king", position: { x: 3, y: 6 }, unstable: false },
    ];
    const hostile = field(0);
    hostile[1][1] = -1;

    const resolved = removeUnrescuedPieces("red", state, new Set(["red-spy"]), hostile);

    expect(resolved.pieces.map((piece) => piece.id)).not.toContain("red-spy");
  });

  it("stable immobile king does not lose", () => {
    const state = kingState();
    state.pieces.push(...Array.from({ length: 8 }, (_, i) => ({ id: `block-${i}`, owner: "blue" as const, type: "pawn" as const, unstable: false, position: { x: 2 + (i % 3), y: 2 + Math.floor(i / 3) } })).filter((piece) => piece.position.x !== 3 || piece.position.y !== 3));
    expect(isKingUnprotected("red", state, field(1))).toBe(false);
  });

  it("a king on hostile territory is unprotected even when an escape exists", () => {
    const state = kingState();
    const hostile = field(-1);
    hostile[3][4] = 0;
    expect(isKingUnprotected("red", state, hostile)).toBe(true);
  });

  it("a king on hostile territory is unprotected with no escape", () => {
    const state = kingState();
    expect(isKingUnprotected("red", state, field(-1))).toBe(true);
  });

  it("an unstable piece without a direct escape still gets its rescue turn", () => {
    const state = createInitialState();
    state.currentPlayer = "red";
    state.pieces = [
      { id: "red-pawn", owner: "red", type: "pawn", position: { x: 0, y: 0 }, unstable: true },
      { id: "blue-king", owner: "blue", type: "king", position: { x: 1, y: 1 }, unstable: false },
    ];
    state.components.red.pawn = [0];
    state.components.blue.king = [0, 1, 1];
    const started = beginTurn(state);

    expect(started.pieces.map((piece) => piece.id)).toContain("red-pawn");
    expect(started.message).toContain("must rescue an unstable round hat");
  });

  it("an opponent piece made unstable by a move survives into its rescue turn", () => {
    const state = createInitialState();
    state.pieces = [
      { id: "blue-king", owner: "blue", type: "king", position: { x: 0, y: 0 }, unstable: false },
      { id: "blue-guard", owner: "blue", type: "pawn", position: { x: 0, y: 1 }, unstable: false },
      { id: "blue-rook", owner: "blue", type: "rook", position: { x: 5, y: 6 }, unstable: false },
      { id: "blue-spy", owner: "blue", type: "spy", position: { x: 1, y: 1 }, unstable: false },
      { id: "red-pawn", owner: "red", type: "pawn", position: { x: 6, y: 6 }, unstable: false },
    ];
    state.components.blue.pawn = [-1];
    state.components.blue.king = [0, 0, 0];
    state.components.blue.rook = [1, 0];
    state.components.blue.spy = [-1, 0, 0];
    state.components.red.pawn = [0];

    const result = applyMove("blue-spy", { x: 5, y: 5 }, state);
    const pawn = result.state.pieces.find((piece) => piece.id === "red-pawn");

    expect(result.ok).toBe(true);
    expect(result.state.currentPlayer).toBe("red");
    expect(pawn).toBeDefined();
    expect(pawn?.unstable).toBe(true);
    expect(result.state.message).toContain("must rescue an unstable round hat");
  });

  it("an unrescued piece still on hostile territory is lost", () => {
    const state = createInitialState();
    state.pieces = [
      { id: "red-pawn-1", owner: "red", type: "pawn", position: { x: 1, y: 1 }, unstable: true },
      { id: "red-pawn-2", owner: "red", type: "pawn", position: { x: 5, y: 5 }, unstable: true },
      { id: "red-king", owner: "red", type: "king", position: { x: 3, y: 6 }, unstable: false },
    ];
    const hostile = field(0);
    hostile[1][1] = -1;
    hostile[5][5] = -1;

    const resolved = removeUnrescuedPieces("red", state, new Set(["red-pawn-2"]), hostile);

    expect(resolved.pieces.map((piece) => piece.id)).toContain("red-pawn-1");
    expect(resolved.pieces.map((piece) => piece.id)).not.toContain("red-pawn-2");
  });

  it("an unrescued piece survives if the move stabilizes its square", () => {
    const state = createInitialState();
    state.pieces = [
      { id: "red-pawn", owner: "red", type: "pawn", position: { x: 1, y: 1 }, unstable: true },
      { id: "red-king", owner: "red", type: "king", position: { x: 3, y: 6 }, unstable: false },
    ];

    const resolved = removeUnrescuedPieces("red", state, new Set(["red-pawn"]), field(0));

    expect(resolved.pieces.map((piece) => piece.id)).toContain("red-pawn");
  });

  it("moving a different piece can rescue an unstable piece by changing the field", () => {
    const state = createInitialState();
    state.pieces = [
      { id: "blue-pawn", owner: "blue", type: "pawn", position: { x: 3, y: 3 }, unstable: true },
      { id: "blue-spy", owner: "blue", type: "spy", position: { x: 2, y: 3 }, unstable: false },
    ];
    state.components.blue.pawn = [0];
    state.components.blue.spy = [1, 0, 0];

    const result = applyMove("blue-spy", { x: 2, y: 2 }, state);
    const pawn = result.state.pieces.find((piece) => piece.id === "blue-pawn");

    expect(result.ok).toBe(true);
    expect(pawn).toBeDefined();
    expect(pawn?.unstable).toBe(false);
  });

  it("moving a different piece removes an unstable piece that remains hostile", () => {
    const state = createInitialState();
    state.pieces = [
      { id: "blue-pawn", owner: "blue", type: "pawn", position: { x: 3, y: 3 }, unstable: true },
      { id: "blue-spy", owner: "blue", type: "spy", position: { x: 2, y: 3 }, unstable: false },
      { id: "red-rook", owner: "red", type: "rook", position: { x: 3, y: 4 }, unstable: false },
    ];
    state.components.blue.pawn = [0];
    state.components.blue.spy = [1, 0, 0];
    state.components.red.rook = [0, 1];

    const result = applyMove("blue-spy", { x: 0, y: 3 }, state);

    expect(result.ok).toBe(true);
    expect(result.state.pieces.some((piece) => piece.id === "blue-pawn")).toBe(false);
    expect(result.state.message).toContain("Blue lost round hat");
  });

  it("uses the visible unstable marker as a rescue deadline", () => {
    const state = createInitialState();
    state.pieces = [
      { id: "blue-pawn", owner: "blue", type: "pawn", position: { x: 3, y: 3 }, unstable: true },
      { id: "blue-spy", owner: "blue", type: "spy", position: { x: 2, y: 3 }, unstable: false },
      { id: "red-pawn", owner: "red", type: "pawn", position: { x: 4, y: 3 }, unstable: false },
      { id: "red-rook", owner: "red", type: "rook", position: { x: 3, y: 4 }, unstable: false },
    ];
    state.components.blue.pawn = [0];
    state.components.blue.spy = [-1, 0, 0];
    state.components.red.pawn = [-1];
    state.components.red.rook = [0, 0];
    state.definitions.spy[0] = structuredClone(state.definitions.pawn[0]);
    expect(getUnstablePieces("blue", state, evaluateField(state)).map((piece) => piece.id)).not.toContain("blue-pawn");

    const result = applyMove("blue-spy", { x: 0, y: 3 }, state);

    expect(result.ok).toBe(true);
    expect(result.state.pieces.some((piece) => piece.id === "blue-pawn")).toBe(false);
  });

  it("moves that leave the moving player's king unprotected are illegal and not offered", () => {
    const state = createInitialState();
    state.pieces = [
      { id: "blue-king", owner: "blue", type: "king", position: { x: 0, y: 0 }, unstable: false },
      { id: "blue-spy", owner: "blue", type: "spy", position: { x: 1, y: 0 }, unstable: false },
      { id: "red-pawn", owner: "red", type: "pawn", position: { x: 0, y: 1 }, unstable: false },
      { id: "red-king", owner: "red", type: "king", position: { x: 6, y: 6 }, unstable: false },
    ];
    state.components.blue.king = [0, 0, 0];
    state.components.blue.spy = [-1, 0, 0];
    state.components.red.pawn = [-1];
    state.components.red.king = [0, 0, 0];
    const result = applyMove("blue-spy", { x: 2, y: 0 }, state);
    expect(result.ok).toBe(false);
    expect(result.reason).toContain("unprotected");
    expect(getPlayableMoves("blue-spy", state)).not.toContainEqual({ x: 2, y: 0 });
  });

  it("focused hint search works without Rust bindings", () => {
    const state = createInitialState();
    state.selectedPieceId = "blue-spy-1";

    const result = applyHintSearch(state, state.selectedPieceId);

    expect(result.ok).toBe(true);
    expect(result.state.selectedPieceId).toBeTruthy();
    expect(result.state.message).toContain("Hint");
  });

  it("a move that leaves the opposing king unprotected gives a rescue turn when possible", () => {
    const state = createInitialState();
    state.pieces = [
      { id: "blue-king", owner: "blue", type: "king", position: { x: 0, y: 0 }, unstable: false },
      { id: "blue-spy", owner: "blue", type: "spy", position: { x: 1, y: 1 }, unstable: false },
      { id: "red-king", owner: "red", type: "king", position: { x: 6, y: 6 }, unstable: false },
    ];
    state.components.blue.king = [0, 0, 0];
    state.components.blue.spy = [-1, 0, 0];
    state.components.red.king = [0, 0, 0];
    state.definitions.spy[0] = structuredClone(state.definitions.pawn[0]);

    const result = applyMove("blue-spy", { x: 0, y: 1 }, state);

    expect(result.ok).toBe(true);
    expect(result.state.status).toBe("playing");
    expect(result.state.currentPlayer).toBe("red");
    expect(result.state.message).toContain("Red Big Hat is in check");

    const hint = findClosestPlayableConfiguration("red", result.state);
    expect(hint).not.toBeNull();
    expect(hint?.changedComponents).toBeGreaterThanOrEqual(0);

    const applied = applyClosestPlayableHint(result.state);
    expect(applied.ok).toBe(true);
    expect(applied.state.selectedPieceId).toBe(hint?.pieceId);
    expect(applied.state.components.red).toEqual(hint?.components);
    expect(applied.state.message).toMatch(/move (round hat|tower|triangle hat|big hat)/);
  });

  it("a checked king with no found rescue loses immediately", () => {
    const state = createInitialState();
    state.currentPlayer = "red";
    state.definitions.pawn[0] = { kind: "preset", name: "Flat", preset: "constant-basin", decayBase: 2, originScale: 1 };
    state.pieces = [
      { id: "red-king", owner: "red", type: "king", position: { x: 3, y: 3 }, unstable: false },
      { id: "blue-pawn-1", owner: "blue", type: "pawn", position: { x: 2, y: 2 }, unstable: false },
      { id: "blue-pawn-2", owner: "blue", type: "pawn", position: { x: 3, y: 2 }, unstable: false },
      { id: "blue-pawn-3", owner: "blue", type: "pawn", position: { x: 4, y: 2 }, unstable: false },
      { id: "blue-pawn-4", owner: "blue", type: "pawn", position: { x: 2, y: 3 }, unstable: false },
      { id: "blue-pawn-5", owner: "blue", type: "pawn", position: { x: 4, y: 3 }, unstable: false },
      { id: "blue-pawn-6", owner: "blue", type: "pawn", position: { x: 2, y: 4 }, unstable: false },
      { id: "blue-pawn-7", owner: "blue", type: "pawn", position: { x: 3, y: 4 }, unstable: false },
      { id: "blue-pawn-8", owner: "blue", type: "pawn", position: { x: 4, y: 4 }, unstable: false },
    ];
    state.components.red.king = [0, 0, 0];
    state.components.blue.pawn = [1];

    const started = beginTurn(state);

    expect(started.status).toBe("blue-won");
    expect(started.currentPlayer).toBe("red");
    expect(started.message).toContain("Red Big Hat is in check");
    expect(started.message).toContain("no legal rescue found");
  });

  it("a player with no legal move loses at turn start", () => {
    const state = createInitialState();
    state.currentPlayer = "blue";
    state.pieces = Array.from({ length: 49 }, (_, index) => ({
      id: index === 24 ? "blue-king" : `blue-pawn-${index}`,
      owner: "blue" as const,
      type: index === 24 ? "king" as const : "pawn" as const,
      position: { x: index % 7, y: Math.floor(index / 7) },
      unstable: false,
    }));

    const started = beginTurn(state);

    expect(started.status).toBe("red-won");
    expect(started.message).toContain("Blue has no legal move");
  });

  it("allows the current player to resign while in check", () => {
    const state = createInitialState();
    state.currentPlayer = "red";
    state.definitions.pawn[0] = { kind: "preset", name: "Flat", preset: "constant-basin", decayBase: 2, originScale: 1 };
    state.pieces = [
      { id: "red-king", owner: "red", type: "king", position: { x: 3, y: 3 }, unstable: true },
      { id: "blue-pawn", owner: "blue", type: "pawn", position: { x: 0, y: 0 }, unstable: false },
    ];
    state.components.red.king = [0, 0, 0];
    state.components.blue.pawn = [1];

    const result = resignInCheck(state);

    expect(result.ok).toBe(true);
    expect(result.state.status).toBe("blue-won");
    expect(result.state.message).toContain("Red resigned while in check");
  });

  it("rejects resignation when the current king is protected", () => {
    const result = resignInCheck(createInitialState());

    expect(result.ok).toBe(false);
    expect(result.reason).toContain("only while your Big Hat is in check");
  });
});
