import { describe, expect, it } from "vitest";
import { playHeuristicTurn } from "../game/ai";
import { createInitialState } from "../game/initialState";
import { applyMove } from "../game/rules";

describe("heuristic opponent", () => {
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

  it("takes a checking move when one is available", () => {
    const state = createInitialState();
    state.currentPlayer = "red";
    state.pieces = [
      { id: "red-king", owner: "red", type: "king", position: { x: 0, y: 0 }, unstable: false },
      { id: "red-spy", owner: "red", type: "spy", position: { x: 1, y: 1 }, unstable: false },
      { id: "blue-king", owner: "blue", type: "king", position: { x: 6, y: 6 }, unstable: false },
    ];
    state.components.red.king = [0, 0, 0];
    state.components.red.spy = [-1, 0, 0];
    state.components.blue.king = [0, 0, 0];

    const result = playHeuristicTurn(state);

    expect(result.status).toBe("playing");
    expect(result.currentPlayer).toBe("blue");
    expect(result.message).toContain("Blue king is in check");
  }, 15_000);
});
