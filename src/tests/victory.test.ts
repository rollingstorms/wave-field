import { describe, expect, it } from "vitest";
import { isKingTrapped, getUnstablePieces, removeUnrescuedPieces, resolveForcedRemovals } from "../game/victory";
import { createInitialState } from "../game/initialState";
import { applyMove } from "../game/rules";
import type { GameState } from "../game/types";

const field = (value = 0) => Array.from({ length: 7 }, () => Array.from({ length: 7 }, () => value));

function kingState(): GameState {
  const state = createInitialState();
  state.pieces = [{ id: "red-king", owner: "red", type: "king", position: { x: 3, y: 3 }, unstable: false }];
  return state;
}

describe("stability and victory", () => {
  it("spies are never unstable", () => {
    const state = createInitialState();
    state.pieces = [{ id: "red-spy", owner: "red", type: "spy", position: { x: 3, y: 3 }, unstable: false }];
    expect(getUnstablePieces("red", state, field(-1))).toHaveLength(0);
  });

  it("stable immobile king does not lose", () => {
    const state = kingState();
    state.pieces.push(...Array.from({ length: 8 }, (_, i) => ({ id: `block-${i}`, owner: "blue" as const, type: "pawn" as const, unstable: false, position: { x: 2 + (i % 3), y: 2 + Math.floor(i / 3) } })).filter((piece) => piece.position.x !== 3 || piece.position.y !== 3));
    expect(isKingTrapped("red", state, field(1))).toBe(false);
  });

  it("unstable king with one stable escape does not lose", () => {
    const state = kingState();
    const hostile = field(-1);
    hostile[3][4] = 0;
    expect(isKingTrapped("red", state, hostile)).toBe(false);
  });

  it("unstable king with no stable escape loses", () => {
    const state = kingState();
    expect(isKingTrapped("red", state, field(-1))).toBe(true);
  });

  it("unstable piece without escape is removed at turn start", () => {
    const state = createInitialState();
    state.pieces = [
      { id: "red-pawn", owner: "red", type: "pawn", position: { x: 0, y: 0 }, unstable: true },
      { id: "blue-king", owner: "blue", type: "king", position: { x: 1, y: 1 }, unstable: false },
    ];
    state.components.red.pawn = [0];
    state.components.blue.king = [1, 0, 0, 0];
    const resolved = resolveForcedRemovals("red", state);
    expect(resolved.pieces.map((piece) => piece.id)).not.toContain("red-pawn");
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

  it("self-trapping moves are illegal", () => {
    const state = createInitialState();
    state.pieces = [
      { id: "blue-king", owner: "blue", type: "king", position: { x: 0, y: 0 }, unstable: false },
      { id: "blue-pawn", owner: "blue", type: "pawn", position: { x: 1, y: 1 }, unstable: false },
      { id: "red-king", owner: "red", type: "king", position: { x: 6, y: 6 }, unstable: false },
    ];
    state.components.blue.king = [-1, 0, 0, 0];
    state.components.blue.pawn = [0];
    state.components.red.king = [1, 0, 0, 0];
    const result = applyMove("blue-pawn", { x: 1, y: 0 }, state);
    expect(result.ok).toBe(false);
  });
});
