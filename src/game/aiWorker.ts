import { playEasyTurn, playHardTurn, playHeuristicTurn } from "./ai";
import { initializeRustEngine } from "./rustEngine";
import type { GameState, Player } from "./types";

interface AiWorkerRequest {
  state: GameState;
  player: Player;
  policy: "easy" | "hard" | "heuristic";
  seed: number;
  variety: number;
}

self.onmessage = async (event: MessageEvent<AiWorkerRequest>) => {
  try {
    const { state, player, policy, seed, variety } = event.data;
    await initializeRustEngine();
    const options = { seed, variety };
    const result = policy === "easy"
      ? playEasyTurn(state, player, options)
      : policy === "hard"
        ? playHardTurn(state, player, { ...options, timeBudgetMs: 2_500 })
        : playHeuristicTurn(state, player, options);
    self.postMessage({ state: result });
  } catch (error) {
    self.postMessage({ error: error instanceof Error ? error.message : String(error) });
  }
};
