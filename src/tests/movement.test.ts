import { describe, expect, it } from "vitest";
import { getLegalMoves } from "../game/movement";
import { createInitialState } from "../game/initialState";
import type { GameState } from "../game/types";

const zeroField = () => Array.from({ length: 7 }, () => Array.from({ length: 7 }, () => 0));

function onePieceState(type: "pawn" | "spy", owner: "red" | "blue" = "red"): GameState {
  const state = createInitialState();
  state.pieces = [{ id: `${owner}-${type}`, owner, type, position: { x: 3, y: 3 }, unstable: false }];
  return state;
}

describe("movement", () => {
  it("red ordinary pieces cannot enter blue territory", () => {
    const state = onePieceState("pawn", "red");
    const field = zeroField();
    field[3][4] = -1;
    expect(getLegalMoves("red-pawn", state, field)).not.toContainEqual({ x: 4, y: 3 });
  });

  it("blue ordinary pieces cannot enter red territory", () => {
    const state = onePieceState("pawn", "blue");
    const field = zeroField();
    field[3][4] = 1;
    expect(getLegalMoves("blue-pawn", state, field)).not.toContainEqual({ x: 4, y: 3 });
  });

  it("both sides can enter neutral territory and move diagonally", () => {
    const state = onePieceState("pawn", "red");
    expect(getLegalMoves("red-pawn", state, zeroField())).toContainEqual({ x: 4, y: 4 });
  });

  it("spies can enter hostile unoccupied territory", () => {
    const state = onePieceState("spy", "red");
    const field = zeroField();
    field[3][4] = -1;
    expect(getLegalMoves("red-spy", state, field)).toContainEqual({ x: 4, y: 3 });
  });

  it("occupied squares block movement", () => {
    const state = onePieceState("pawn", "red");
    state.pieces.push({ id: "blue-pawn", owner: "blue", type: "pawn", position: { x: 4, y: 3 }, unstable: false });
    expect(getLegalMoves("red-pawn", state, zeroField())).not.toContainEqual({ x: 4, y: 3 });
  });
});
