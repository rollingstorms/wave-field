import { BIG_BOARD_SIZE, BOARD_SIZE } from "./constants";
import type { Coefficient, GameState, MoveResult, PieceType, Player, Position } from "./types";

interface RustBindings {
  default: () => Promise<unknown>;
  apply_closest_playable_hint_json: (state: string) => string;
  apply_move_json: (pieceId: string, x: number, y: number, state: string, analyzeCheckmate: boolean) => string;
  apply_tuning_json: (player: Player, pieceType: PieceType, componentIndex: number, value: Coefficient, state: string) => string;
  begin_turn_json: (state: string, analyzeCheckmate: boolean) => string;
  closest_playable_configuration_json: (player: Player, state: string) => string;
  hint_search_json: (player: Player, focusedPieceId: string, state: string, maxTuningStates: number, timeBudgetMs: number) => string;
  evaluate_field_json: (state: string) => string;
  king_unprotected_json: (player: Player, state: string) => boolean;
  legal_moves_json: (pieceId: string, state: string) => string;
  mark_instability_json: (state: string) => string;
  playable_moves_json: (pieceId: string, state: string) => string;
  play_easy_turn_json: (player: Player, state: string, seed: number, variety: number, timeBudgetMs: number) => string;
  play_heuristic_turn_json: (player: Player, state: string, seed: number, variety: number, timeBudgetMs: number) => string;
  randomize_tuning_json: (rolls: string, state: string) => string;
  reset_tuning_json: (state: string) => string;
  resign_in_check_json: (state: string) => string;
  unstable_piece_ids_json: (player: Player, state: string) => string;
}

let bindings: RustBindings | null = null;

function requestedRuleEngine(): "rust" | "ts" {
  if (!import.meta.env.DEV) return "ts";
  if (BOARD_SIZE === BIG_BOARD_SIZE) return "ts";
  const requested = new URLSearchParams(globalThis.location?.search ?? "").get("engine");
  if (requested === "ts" || requested === "typescript") return "ts";
  return "rust";
}

export async function initializeRustEngine(): Promise<void> {
  if (!import.meta.env.DEV || bindings) return;
  if (requestedRuleEngine() === "ts") {
    document.documentElement.dataset.ruleEngine = "typescript";
    console.info("Wave Field: TypeScript rule engine active");
    return;
  }
  const modulePath = "/engine/pkg/wave_field_engine.js";
  const loaded = await import(/* @vite-ignore */ modulePath) as RustBindings;
  await loaded.default();
  bindings = loaded;
  document.documentElement.dataset.ruleEngine = "rust-wasm";
  console.info("Wave Field: Rust rule engine active");
}

export function rustEngineActive(): boolean {
  return bindings !== null;
}

function stateJson(state: GameState): string {
  return JSON.stringify(state);
}

function callRust<T>(operation: () => T): T | null {
  if (!bindings) return null;
  try {
    return operation();
  } catch (error) {
    console.warn("Wave Field: Rust rule engine failed; falling back to TypeScript.", error);
    bindings = null;
    document.documentElement.dataset.ruleEngine = "typescript";
    return null;
  }
}

export function rustEvaluateField(state: GameState): number[][] | null {
  return callRust(() => JSON.parse(bindings!.evaluate_field_json(stateJson(state))) as number[][]);
}

export function rustLegalMoves(pieceId: string, state: GameState): Position[] | null {
  return callRust(() => JSON.parse(bindings!.legal_moves_json(pieceId, stateJson(state))) as Position[]);
}

export function rustPlayableMoves(pieceId: string, state: GameState): Position[] | null {
  return callRust(() => JSON.parse(bindings!.playable_moves_json(pieceId, stateJson(state))) as Position[]);
}

export function rustClosestPlayableConfiguration<T>(player: Player, state: GameState): T | null {
  return callRust(() => JSON.parse(bindings!.closest_playable_configuration_json(player, stateJson(state))) as T);
}

export function rustHintSearch<T>(
  player: Player,
  focusedPieceId: string | null,
  state: GameState,
  maxTuningStates: number,
  timeBudgetMs: number,
): T | null {
  return callRust(() =>
    JSON.parse(bindings!.hint_search_json(player, focusedPieceId ?? "", stateJson(state), maxTuningStates, timeBudgetMs)) as T);
}

export function rustApplyMove(
  pieceId: string,
  destination: Position,
  state: GameState,
  analyzeCheckmate: boolean,
): MoveResult | null {
  return callRust(() =>
    JSON.parse(bindings!.apply_move_json(pieceId, destination.x, destination.y, stateJson(state), analyzeCheckmate)) as MoveResult);
}

export function rustBeginTurn(state: GameState, analyzeCheckmate: boolean): GameState | null {
  return callRust(() => JSON.parse(bindings!.begin_turn_json(stateJson(state), analyzeCheckmate)) as GameState);
}

export function rustApplyTuning(
  player: Player,
  pieceType: PieceType,
  componentIndex: number,
  value: Coefficient,
  state: GameState,
): MoveResult | null {
  return callRust(() =>
    JSON.parse(bindings!.apply_tuning_json(player, pieceType, componentIndex, value, stateJson(state))) as MoveResult);
}

export function rustResignInCheck(state: GameState): MoveResult | null {
  return callRust(() => JSON.parse(bindings!.resign_in_check_json(stateJson(state))) as MoveResult);
}

export function rustApplyClosestPlayableHint(state: GameState): MoveResult | null {
  return callRust(() => JSON.parse(bindings!.apply_closest_playable_hint_json(stateJson(state))) as MoveResult);
}

export function rustResetTuning(state: GameState): MoveResult | null {
  return callRust(() => JSON.parse(bindings!.reset_tuning_json(stateJson(state))) as MoveResult);
}

export function rustRandomizeTuning(state: GameState, rolls: number[]): MoveResult | null {
  return callRust(() => JSON.parse(bindings!.randomize_tuning_json(JSON.stringify(rolls), stateJson(state))) as MoveResult);
}

export function rustUnstablePieceIds(player: Player, state: GameState): string[] | null {
  return callRust(() => JSON.parse(bindings!.unstable_piece_ids_json(player, stateJson(state))) as string[]);
}

export function rustKingUnprotected(player: Player, state: GameState): boolean | null {
  return callRust(() => bindings!.king_unprotected_json(player, stateJson(state)));
}

export function rustMarkInstability(state: GameState): GameState | null {
  return bindings
    ? JSON.parse(bindings.mark_instability_json(stateJson(state))) as GameState
    : null;
}

export function rustPlayHeuristicTurn(
  state: GameState,
  player: Player,
  seed: number,
  variety: number,
  timeBudgetMs: number,
): GameState | null {
  return bindings
    ? JSON.parse(bindings.play_heuristic_turn_json(player, stateJson(state), seed, variety, timeBudgetMs)) as GameState
    : null;
}

export function rustPlayEasyTurn(
  state: GameState,
  player: Player,
  seed: number,
  variety: number,
  timeBudgetMs: number,
): GameState | null {
  return callRust(() =>
    JSON.parse(bindings!.play_easy_turn_json(player, stateJson(state), seed, variety, timeBudgetMs)) as GameState);
}
