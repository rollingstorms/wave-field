import { spawnSync } from "node:child_process";
import { describe, expect, it } from "vitest";
import { evaluateField } from "../field/evaluateField";
import { playHeuristicTurn } from "../game/ai";
import { createInitialState } from "../game/initialState";
import { getLegalMoves } from "../game/movement";
import {
  applyClosestPlayableHint,
  applyMove,
  applyTuning,
  beginTurn,
  getPlayableMoves,
  randomizeTuning,
  resetTuning,
  resignInCheck,
} from "../game/rules";
import type { GameState, PieceType, Player, Position } from "../game/types";
import { getUnstablePieces, isKingUnprotected } from "../game/victory";

const engineBinary = "engine/target/debug/wave-field-engine";

function rust<T>(method: string, state: GameState, params: Record<string, unknown> = {}): T {
  const result = spawnSync(engineBinary, {
    input: `${JSON.stringify({ method, state, ...params })}\n`,
    encoding: "utf8",
  });
  if (result.status !== 0) {
    throw new Error(`Rust engine failed:\n${result.stderr}`);
  }
  return JSON.parse(result.stdout.trim()) as T;
}

describe("Rust engine parity", () => {
  it("matches the complete initial field", () => {
    const state = createInitialState();
    expect(rust<number[][]>("evaluateField", state)).toEqual(evaluateField(state));
  });

  it("matches legal and playable moves for every initial piece", () => {
    const state = createInitialState();
    const field = evaluateField(state);
    for (const piece of state.pieces) {
      expect(rust<Position[]>("legalMoves", state, { pieceId: piece.id }))
        .toEqual(getLegalMoves(piece.id, state, field));
      if (piece.owner === state.currentPlayer) {
        expect(rust<Position[]>("playableMoves", state, { pieceId: piece.id }))
          .toEqual(getPlayableMoves(piece.id, state, field));
      }
    }
  });

  it("matches move resolution, loss timing, history, and messages", () => {
    const state = createInitialState();
    const destination = getPlayableMoves("blue-spy-1", state)[0];
    expect(rust("applyMove", state, { pieceId: "blue-spy-1", destination }))
      .toEqual(applyMove("blue-spy-1", destination, state));
  });

  it("matches tuning activation-order behavior", () => {
    let state = createInitialState();
    const actions: Array<[PieceType, number, -1 | 1]> = [
      ["king", 0, -1],
      ["king", 2, -1],
      ["spy", 2, 1],
    ];
    for (const [pieceType, componentIndex, value] of actions) {
      const expected = applyTuning("blue", pieceType, componentIndex, value, state);
      const actual = rust<typeof expected>("applyTuning", state, {
        player: "blue",
        pieceType,
        componentIndex,
        value,
      });
      expect(actual).toEqual(expected);
      state = expected.state;
    }
  });

  it("matches randomize and reset tuning actions", () => {
    const state = createInitialState();
    const rolls = [0, 0.3, 0.6, 0.9];
    const queue = [...rolls];
    const expectedRandom = randomizeTuning(state, () => queue.shift() ?? 0);
    const actualRandom = rust<typeof expectedRandom>("randomizeTuning", state, { rolls });
    expect(actualRandom).toEqual(expectedRandom);
    expect(rust("resetTuning", expectedRandom.state)).toEqual(resetTuning(expectedRandom.state));
  });

  it("matches instability, check, and begin-turn behavior on custom states", () => {
    const state = createInitialState();
    state.currentPlayer = "red";
    state.pieces = [
      { id: "red-king", owner: "red", type: "king", position: { x: 3, y: 3 }, unstable: false },
      { id: "red-spy", owner: "red", type: "spy", position: { x: 2, y: 3 }, unstable: false },
      { id: "blue-pawn", owner: "blue", type: "pawn", position: { x: 3, y: 2 }, unstable: false },
    ];
    state.components.red.king = [0, 0, 0];
    state.components.blue.pawn = [-1];
    const field = evaluateField(state);
    for (const player of ["red", "blue"] as Player[]) {
      expect(rust<string[]>("unstablePieceIds", state, { player }))
        .toEqual(getUnstablePieces(player, state, field).map((piece) => piece.id));
      expect(rust<boolean>("kingUnprotected", state, { player }))
        .toBe(isKingUnprotected(player, state, field));
    }
    expect(rust("beginTurn", state, { analyzeCheckmate: false }))
      .toEqual(beginTurn(state, { analyzeCheckmate: false }));
  });

  it("matches hint and resignation behavior while in check", () => {
    const state = createInitialState();
    state.currentPlayer = "red";
    state.pieces = [
      { id: "red-king", owner: "red", type: "king", position: { x: 6, y: 6 }, unstable: false },
      { id: "blue-spy", owner: "blue", type: "spy", position: { x: 0, y: 1 }, unstable: false },
    ];
    state.components.red.king = [0, 0, 0];
    state.components.blue.spy = [-1, 0, 0];

    expect(rust("applyClosestPlayableHint", state)).toEqual(applyClosestPlayableHint(state));
    expect(rust("resignInCheck", state)).toEqual(resignInCheck(state));
  });

  it("matches heuristic AI turns", () => {
    const opening = createInitialState();
    expect(rust("playHeuristicTurn", opening, { player: "blue", seed: 17, variety: 0, timeBudgetMs: 1_000 }))
      .toEqual(playHeuristicTurn(opening, "blue", { seed: 17, variety: 0, timeBudgetMs: 1_000 }));

  }, 15_000);
});
