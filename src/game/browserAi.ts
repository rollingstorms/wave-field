import type { GameState, Player } from "./types";

export function requestBrowserAiTurn(
  state: GameState,
  player: Player,
  policy: "easy" | "hard" | "heuristic",
  seed: number,
  variety: number,
): Promise<GameState> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./aiWorker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (event: MessageEvent<{ state?: GameState; error?: string }>) => {
      worker.terminate();
      if (event.data.state) resolve(event.data.state);
      else reject(new Error(event.data.error ?? "AI worker failed"));
    };
    worker.onerror = (event) => {
      worker.terminate();
      reject(new Error(event.message || "AI worker failed"));
    };
    worker.postMessage({ state, player, policy, seed, variety });
  });
}
