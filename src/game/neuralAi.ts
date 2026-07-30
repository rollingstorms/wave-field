import type { Coefficient, GameState, PieceType, Position } from "./types";

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
  type?: "move";
  pieceId: string;
  destination: Position;
}

export interface NeuralTune {
  type: "tune";
  pieceType: PieceType;
  componentIndex: number;
  value: Exclude<Coefficient, 0>;
}

export type NeuralTurnAction = NeuralMove | NeuralTune;

interface NeuralMoveResponse {
  ok: boolean;
  action?: NeuralMove;
  actions?: NeuralTurnAction[];
  error?: string;
}

export async function requestNeuralMove(
  state: GameState,
  policy: NeuralPolicy = "neural-residual",
): Promise<NeuralMove> {
  const actions = await requestNeuralTurn(state, policy);
  const move = actions.find((action): action is NeuralMove => action.type !== "tune");
  if (!move) throw new Error("Neural model server did not return a move");
  return move;
}

export async function requestNeuralTurn(
  state: GameState,
  policy: NeuralPolicy = "neural-residual",
): Promise<NeuralTurnAction[]> {
  const endpoint = neuralEndpoints[policy];
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ state }),
  });
  const payload = await response.json() as NeuralMoveResponse;
  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || `Neural model server returned ${response.status}`);
  }
  if (payload.actions?.length) return payload.actions;
  if (payload.action) return [{ type: "move", ...payload.action }];
  throw new Error("Neural model server did not return an action");
}
