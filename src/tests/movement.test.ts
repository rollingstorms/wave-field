import { describe, expect, it } from "vitest";
import { getLegalMoves } from "../game/movement";
import { createInitialState } from "../game/initialState";
import type { GameState } from "../game/types";

const zeroField = () => Array.from({ length: 7 }, () => Array.from({ length: 7 }, () => 0));

function onePieceState(type: "pawn" | "rook" | "spy" | "king", owner: "red" | "blue" = "red"): GameState {
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

  it("pieces can slide any distance along a neutral diagonal", () => {
    const state = onePieceState("pawn", "red");
    expect(getLegalMoves("red-pawn", state, zeroField())).toContainEqual({ x: 6, y: 6 });
  });

  it("spies ignore hostile territory along their entire ray", () => {
    const state = onePieceState("spy", "red");
    const field = zeroField();
    field[3][4] = -1;
    field[3][5] = -1;
    const moves = getLegalMoves("red-spy", state, field);
    expect(moves).toContainEqual({ x: 4, y: 3 });
    expect(moves).toContainEqual({ x: 5, y: 3 });
    expect(moves).toContainEqual({ x: 6, y: 3 });
  });

  it("spies still stop at the first occupied square", () => {
    const state = onePieceState("spy", "red");
    state.pieces.push({ id: "blue-pawn", owner: "blue", type: "pawn", position: { x: 5, y: 3 }, unstable: false });
    const field = zeroField();
    field[3][4] = -1;
    field[3][5] = -1;
    field[3][6] = -1;
    const moves = getLegalMoves("red-spy", state, field);

    expect(moves).toContainEqual({ x: 4, y: 3 });
    expect(moves).not.toContainEqual({ x: 5, y: 3 });
    expect(moves).not.toContainEqual({ x: 6, y: 3 });
  });

  it("occupied squares block themselves and every square beyond them", () => {
    const state = onePieceState("pawn", "red");
    state.pieces.push({ id: "blue-pawn", owner: "blue", type: "pawn", position: { x: 4, y: 3 }, unstable: false });
    expect(getLegalMoves("red-pawn", state, zeroField())).not.toContainEqual({ x: 4, y: 3 });
    expect(getLegalMoves("red-pawn", state, zeroField())).not.toContainEqual({ x: 5, y: 3 });
  });

  it("hostile territory blocks every square beyond it on the ray", () => {
    const state = onePieceState("rook", "red");
    const field = zeroField();
    field[3][5] = -1;
    const moves = getLegalMoves("red-rook", state, field);
    expect(moves).toContainEqual({ x: 4, y: 3 });
    expect(moves).not.toContainEqual({ x: 5, y: 3 });
    expect(moves).not.toContainEqual({ x: 6, y: 3 });
  });

  it("pieces cannot turn while moving", () => {
    const state = onePieceState("king", "red");
    expect(getLegalMoves("red-king", state, zeroField())).not.toContainEqual({ x: 5, y: 4 });
  });
});
