import type { GameState, Position } from "./types";

export type NeuralPolicy = "neural-residual" | "neural-transformer";
export type AiPolicy = "heuristic" | NeuralPolicy;

const neuralEndpoints: Record<NeuralPolicy, string> = {
  "neural-residual": "http://127.0.0.1:8765/move",
  "neural-transformer": "http://127.0.0.1:8766/move",
};

export function isNeuralPolicy(policy: AiPolicy): policy is NeuralPolicy {
  return policy.startsWith("neural-");
}

export function policyLabel(policy: AiPolicy | "human"): string {
  switch (policy) {
    case "human":
      return "Human";
    case "heuristic":
      return "Heuristic";
    case "neural-residual":
      return "Neural residual";
    case "neural-transformer":
      return "Neural transformer";
  }
}

export interface NeuralMove {
  pieceId: string;
  destination: Position;
}

interface NeuralMoveResponse {
  ok: boolean;
  action?: NeuralMove;
  error?: string;
}

export async function requestNeuralMove(
  state: GameState,
  policy: NeuralPolicy = "neural-residual",
): Promise<NeuralMove> {
  const endpoint = neuralEndpoints[policy];
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ state }),
  });
  const payload = await response.json() as NeuralMoveResponse;
  if (!response.ok || !payload.ok || !payload.action) {
    throw new Error(payload.error || `Neural model server returned ${response.status}`);
  }
  return payload.action;
}
