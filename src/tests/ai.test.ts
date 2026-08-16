import { describe, expect, it } from "vitest";
import { playEasyTurn, playHardTurn, playHeuristicTurn } from "../game/ai";
import { createInitialState, snapshot } from "../game/initialState";
import { applyMove } from "../game/rules";
import type { GameState, PieceType } from "../game/types";

const pieceTypes: PieceType[] = ["pawn", "rook", "spy", "king"];

function zeroComponents(state: GameState) {
  for (const player of ["blue", "red"] as const) {
    for (const pieceType of pieceTypes) {
      state.components[player][pieceType] = state.components[player][pieceType].map(() => 0) as never;
      state.activationOrders[player][pieceType] = [];
    }
  }
}

function setPiecePosition(state: GameState, pieceId: string, x: number, y: number) {
  state.pieces = state.pieces.map((piece) => piece.id === pieceId ? { ...piece, position: { x, y } } : piece);
}

describe("heuristic opponent", () => {
  it("easy minimax completes a legal blue turn without tuning", () => {
    const opening = createInitialState();

    const result = playEasyTurn(opening, "blue", { seed: 7 });

    expect(result.status).toBe("playing");
    expect(result.currentPlayer).toBe("red");
    expect(result.components.blue).toEqual(opening.components.blue);
    expect(result.history).toHaveLength(opening.history.length + 1);
  });

  it("completes a legal red turn", () => {
    const opening = createInitialState();
    const blueMove = applyMove("blue-pawn-1", { x: 1, y: 1 }, opening);
    expect(blueMove.ok).toBe(true);

    const result = playHeuristicTurn(blueMove.state);

    expect(result.currentPlayer === "blue" || result.status === "red-won").toBe(true);
    expect(result.history).toHaveLength(blueMove.state.history.length + 1);
  });

  it("completes a legal blue turn", () => {
    const opening = createInitialState();

    const result = playHeuristicTurn(opening, "blue");

    expect(result.currentPlayer === "red" || result.status === "blue-won").toBe(true);
    expect(result.history).toHaveLength(opening.history.length + 1);
  });

  it("hard opponent completes a legal blue turn", () => {
    const opening = createInitialState();

    const result = playHardTurn(opening, "blue", { timeBudgetMs: 50 });

    expect(result.currentPlayer === "red" || result.status === "blue-won").toBe(true);
    expect(result.history).toHaveLength(opening.history.length + 1);
  });

  it("ends the game when the AI has no legal move", () => {
    const state = createInitialState();
    state.currentPlayer = "blue";
    state.pieces = [
      { id: "red-king", owner: "red", type: "king", position: { x: 6, y: 6 }, unstable: false },
    ];
    state.components.red.king = [0, 0, 0];

    const result = playHeuristicTurn(state, "blue");

    expect(result.status).toBe("red-won");
    expect(result.message).toContain("Blue has no legal move");
  });

  it("avoids immediately moving the same piece back when alternatives exist", () => {
    const beforeRed = createInitialState();
    beforeRed.currentPlayer = "red";
    beforeRed.pieces = [
      { id: "red-king", owner: "red", type: "king", position: { x: 0, y: 0 }, unstable: false },
      { id: "red-rook-1", owner: "red", type: "rook", position: { x: 3, y: 3 }, unstable: false },
      { id: "blue-king", owner: "blue", type: "king", position: { x: 6, y: 6 }, unstable: false },
    ];
    zeroComponents(beforeRed);

    const afterRed = structuredClone(beforeRed);
    afterRed.currentPlayer = "blue";
    setPiecePosition(afterRed, "red-rook-1", 4, 4);

    const beforeBlue = structuredClone(afterRed);
    const current = structuredClone(afterRed);
    current.currentPlayer = "red";
    current.history = [snapshot(beforeRed), snapshot(afterRed), snapshot(beforeBlue)];
    current.turnNumber = 2;

    const result = playHeuristicTurn(current, "red", { timeBudgetMs: 1_000 });
    const rook = result.pieces.find((piece) => piece.id === "red-rook-1");

    expect(rook?.position).not.toEqual({ x: 3, y: 3 });
  });

  it("uses seeded stochasticity to diversify repeated board positions reproducibly", () => {
    const state = createInitialState();
    state.currentPlayer = "red";
    state.pieces = [
      { id: "red-king", owner: "red", type: "king", position: { x: 0, y: 0 }, unstable: false },
      { id: "red-rook", owner: "red", type: "rook", position: { x: 3, y: 3 }, unstable: false },
      { id: "blue-king", owner: "blue", type: "king", position: { x: 6, y: 6 }, unstable: false },
    ];
    zeroComponents(state);
    state.history = [snapshot(state), snapshot(state), snapshot(state)];

    const signature = (result: GameState) => result.pieces
      .map((piece) => `${piece.id}:${piece.position.x},${piece.position.y}`)
      .join("|");
    const first = playHeuristicTurn(structuredClone(state), "red", { seed: 17, variety: 0.9, timeBudgetMs: 1_000 });
    const repeated = playHeuristicTurn(structuredClone(state), "red", { seed: 17, variety: 0.9, timeBudgetMs: 1_000 });
    const outcomes = new Set([17, 29, 41, 53].map((seed) =>
      signature(playHeuristicTurn(structuredClone(state), "red", { seed, variety: 0.9, timeBudgetMs: 1_000 })),
    ));

    expect(signature(repeated)).toBe(signature(first));
    expect(outcomes.size).toBeGreaterThan(1);
  }, 15_000);

  it("takes a checking move when one is available", () => {
    const state = createInitialState();
    state.currentPlayer = "red";
    state.pieces = [
      { id: "red-king", owner: "red", type: "king", position: { x: 0, y: 0 }, unstable: false },
      { id: "red-spy", owner: "red", type: "spy", position: { x: 2, y: 2 }, unstable: false },
      { id: "blue-king", owner: "blue", type: "king", position: { x: 6, y: 6 }, unstable: false },
    ];
    state.components.red.king = [0, 0, 0];
    state.components.red.spy = [-1, 0, 0];
    state.components.blue.king = [0, 0, 0];

    const result = playHeuristicTurn(state);

    expect(result.status).toBe("playing");
    expect(result.currentPlayer).toBe("blue");
    expect(result.message).toContain("Blue Big Hat is in check");
  }, 15_000);
});
