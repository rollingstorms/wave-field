import { evaluateField } from "../field/evaluateField";
import { isSquareCompatible } from "../field/projection";
import { COMPONENT_COUNTS, TUNING_STRENGTH } from "./constants";
import { snapshot } from "./initialState";
import { getLegalMoves } from "./movement";
import { applyMove, opponent } from "./rules";
import { activationOrderForProfile } from "./tuning";
import type { Coefficient, GameSnapshot, GameState, Piece, PieceType, Player, PlayerComponents, Position } from "./types";
import { getUnstablePieces, isKingUnprotected, markInstability } from "./victory";

const pieceTypes: PieceType[] = ["pawn", "rook", "spy", "king"];
const coefficients: Coefficient[] = [-1, 0, 1];
const materialValue = { pawn: 2, rook: 4, spy: 3, king: 100 } as const;
const exactCandidateLimit = 6;
const defaultTimeBudgetMs = 180;
const fullAnalysisLimit = 3;
const repetitionLookback = 18;
const repeatedStatePenalty = 900;
const repeatedBoardPenalty = 320;
const immediateReversalPenalty = 500;

export interface AiTurnOptions {
  seed?: number;
  variety?: number;
  timeBudgetMs?: number;
  debug?: boolean;
}

function nowMs(): number {
  return globalThis.performance?.now() ?? Date.now();
}

function addTuningCandidate(candidates: PlayerComponents[], seen: Set<string>, candidate: PlayerComponents) {
  const key = JSON.stringify(candidate);
  if (!seen.has(key)) {
    seen.add(key);
    candidates.push(candidate);
  }
}

function profileAfterControlChange(
  state: GameState,
  player: Player,
  pieceType: PieceType,
  componentIndex: number,
  value: Exclude<Coefficient, 0>,
): Coefficient[] | null {
  const profile = [...state.components[player][pieceType]] as Coefficient[];
  const currentValue = profile[componentIndex];
  if (currentValue === value) return null;

  const activeIndices = profile.flatMap((coefficient, index) => coefficient === 0 ? [] : [index]);
  const nextOrder = state.activationOrders[player][pieceType]
    .filter((index) => activeIndices.includes(index) && index !== componentIndex);
  for (const index of activeIndices) {
    if (index !== componentIndex && !nextOrder.includes(index)) nextOrder.push(index);
  }

  if (currentValue === 0 && activeIndices.length >= TUNING_STRENGTH[pieceType]) {
    const evictedIndex = nextOrder.shift();
    if (evictedIndex !== undefined) profile[evictedIndex] = 0;
  }

  profile[componentIndex] = value;
  return profile;
}

function tuningCandidates(state: GameState, player: Player, maxCandidates = 28): PlayerComponents[] {
  const current = state.components[player];
  const candidates = [structuredClone(current)];
  const seen = new Set([JSON.stringify(current)]);

  for (const pieceType of pieceTypes) {
    const profile = current[pieceType];
    for (let index = 0; index < COMPONENT_COUNTS[pieceType]; index += 1) {
      for (const value of coefficients.filter((coefficient) => coefficient !== 0)) {
        const nextProfile = profileAfterControlChange(state, player, pieceType, index, value);
        if (!nextProfile) continue;

        const next = structuredClone(current);
        next[pieceType] = nextProfile as never;
        addTuningCandidate(candidates, seen, next);
        if (candidates.length >= maxCandidates) return candidates;
      }
    }
  }
  return candidates;
}

function scoreState(state: GameState, player: Player, field: number[][]): number {
  if (state.status === `${player}-won`) return 1_000_000;
  if (state.status !== "playing") return -1_000_000;

  const enemy = player === "red" ? "blue" : "red";
  let score = 0;

  for (const piece of state.pieces) {
    const direction = piece.owner === player ? 1 : -1;
    score += direction * materialValue[piece.type] * 120;
    const centerDistance = Math.abs(piece.position.x - 3) + Math.abs(piece.position.y - 3);
    score += direction * (6 - centerDistance) * (piece.type === "spy" ? 2 : 1);
    score += direction * getLegalMoves(piece.id, state, field).length * 1.5;
  }

  for (const row of field) {
    for (const value of row) {
      if (value > 0) score += player === "red" ? 2 : -2;
      if (value < 0) score += player === "blue" ? 2 : -2;
    }
  }

  score -= getUnstablePieces(player, state, field).filter((piece) => piece.type !== "king").length * 90;
  score += getUnstablePieces(enemy, state, field).filter((piece) => piece.type !== "king").length * 45;

  const ownKing = state.pieces.find((piece) => piece.owner === player && piece.type === "king");
  if (ownKing) {
    const value = field[ownKing.position.y][ownKing.position.x];
    score += isSquareCompatible(player, value) ? Math.min(Math.abs(value), 4) * 25 : -10_000;
  }
  if (isKingUnprotected(enemy, state, field)) score += 400_000;

  return score;
}

function playerName(player: Player): string {
  return player === "red" ? "Red" : "Blue";
}

function winStatus(player: Player) {
  return player === "red" ? "red-won" as const : "blue-won" as const;
}

function hashUnit(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) / 4294967295;
}

function samePosition(left: Position, right: Position): boolean {
  return left.x === right.x && left.y === right.y;
}

function stateKey(state: GameState | GameSnapshot): string {
  const pieces = [...state.pieces]
    .sort((left, right) => left.id.localeCompare(right.id))
    .map((piece) => `${piece.id}:${piece.owner}:${piece.type}:${piece.position.x},${piece.position.y}`)
    .join("|");
  const components = (["blue", "red"] as const)
    .map((side) => `${side}:${pieceTypes.map((pieceType) => state.components[side][pieceType].join(",")).join("/")}`)
    .join("|");
  return `${state.currentPlayer}|${state.status}|${pieces}|${components}`;
}

function boardPositionKey(state: GameState | GameSnapshot): string {
  const pieces = [...state.pieces]
    .sort((left, right) => left.id.localeCompare(right.id))
    .map((piece) => `${piece.id}:${piece.position.x},${piece.position.y}`)
    .join("|");
  return `${state.currentPlayer}|${state.status}|${pieces}`;
}

function recentCounts(state: GameState, keyFor: (entry: GameState | GameSnapshot) => string): Map<string, number> {
  const counts = new Map<string, number>();
  for (const entry of state.history.slice(-repetitionLookback)) {
    const key = keyFor(entry);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  const currentKey = keyFor(state);
  counts.set(currentKey, (counts.get(currentKey) ?? 0) + 1);
  return counts;
}

function movedPiece(before: GameSnapshot, after: GameSnapshot): { pieceId: string; from: Position; to: Position } | null {
  for (const piece of before.pieces) {
    const next = after.pieces.find((candidate) => candidate.id === piece.id);
    if (next && !samePosition(piece.position, next.position)) {
      return { pieceId: piece.id, from: piece.position, to: next.position };
    }
  }
  return null;
}

function lastMoveByPiece(state: GameState, player: Player): Map<string, { from: Position; to: Position }> {
  const timeline: GameSnapshot[] = [...state.history, snapshot(state)];
  const moves = new Map<string, { from: Position; to: Position }>();
  for (let index = timeline.length - 2; index >= 0; index -= 1) {
    const before = timeline[index];
    if (before.currentPlayer !== player) continue;
    const move = movedPiece(before, timeline[index + 1]);
    if (move && !moves.has(move.pieceId)) moves.set(move.pieceId, { from: move.from, to: move.to });
  }
  return moves;
}

function loopPenalty(
  preview: GameState,
  piece: Piece,
  destination: Position,
  repetitionCounts: ReadonlyMap<string, number>,
  boardPositionCounts: ReadonlyMap<string, number>,
  recentMoves: ReadonlyMap<string, { from: Position; to: Position }>,
): number {
  const repeatCount = repetitionCounts.get(stateKey(preview)) ?? 0;
  const boardRepeatCount = boardPositionCounts.get(boardPositionKey(preview)) ?? 0;
  const previousMove = recentMoves.get(piece.id);
  const reversesLastMove = Boolean(
    previousMove
    && samePosition(previousMove.to, piece.position)
    && samePosition(previousMove.from, destination),
  );
  return repeatCount * repeatedStatePenalty
    + boardRepeatCount * repeatedBoardPenalty
    + (reversesLastMove ? immediateReversalPenalty : 0);
}

export function playHeuristicTurn(state: GameState, player: Player = "red", options: AiTurnOptions = {}): GameState {
  if (state.status !== "playing" || state.currentPlayer !== player) return state;

  const choices: Array<{ tuned: GameState; pieceId: string; destination: { x: number; y: number }; preview: GameState; score: number }> = [];
  const fieldCache = new WeakMap<GameState, number[][]>();
  const repetitionCounts = recentCounts(state, stateKey);
  const boardPositionCounts = recentCounts(state, boardPositionKey);
  const currentBoardVisits = boardPositionCounts.get(boardPositionKey(state)) ?? 1;
  const recentMoves = lastMoveByPiece(state, player);
  const startedAt = nowMs();
  const deadline = startedAt + Math.max(20, options.timeBudgetMs ?? defaultTimeBudgetMs);
  let profilesChecked = 0;
  let movesChecked = 0;
  let bestScore = Number.NEGATIVE_INFINITY;

  function getField(candidate: GameState): number[][] {
    const cached = fieldCache.get(candidate);
    if (cached) return cached;
    const field = evaluateField(candidate);
    fieldCache.set(candidate, field);
    return field;
  }

  function rememberChoice(tuned: GameState, pieceId: string, destination: { x: number; y: number }, preview: GameState, score: number) {
    if (choices.length < exactCandidateLimit || score > bestScore) {
      choices.push({ tuned, pieceId, destination, preview, score });
      choices.sort((a, b) => b.score - a.score);
      choices.length = Math.min(choices.length, exactCandidateLimit);
      bestScore = choices.at(-1)?.score ?? Number.NEGATIVE_INFINITY;
    }
  }

  for (const profile of tuningCandidates(state, player)) {
    profilesChecked += 1;
    const components = structuredClone(state.components);
    components[player] = profile;
    const activationOrders = structuredClone(state.activationOrders);
    activationOrders[player] = activationOrderForProfile(profile);
    const tunedBase = { ...state, components, activationOrders, selectedPieceId: null };
    const tunedBaseField = getField(tunedBase);
    const tuned = markInstability(
      tunedBase,
      tunedBaseField,
    );
    fieldCache.set(tuned, tunedBaseField);

    for (const piece of tuned.pieces.filter((candidate) => candidate.owner === player)) {
      for (const destination of getLegalMoves(piece.id, tuned, tunedBaseField)) {
        movesChecked += 1;
        const result = applyMove(piece.id, destination, tuned, { analyzeCheckmate: false });
        if (!result.ok) continue;
        const score = scoreState(result.state, player, getField(result.state))
          - loopPenalty(result.state, piece, destination, repetitionCounts, boardPositionCounts, recentMoves);
        rememberChoice(tuned, piece.id, destination, result.state, score);
        if (nowMs() >= deadline && choices.length > 0) break;
      }
      if (nowMs() >= deadline && choices.length > 0) break;
    }
    if (nowMs() >= deadline && choices.length > 0) break;
  }

  if (options.debug) {
    console.debug("AI turn search", {
      player,
      profilesChecked,
      movesChecked,
      choices: choices.length,
      elapsedMs: Math.round(nowMs() - startedAt),
      budgetMs: Math.round(deadline - startedAt),
      currentBoardVisits,
    });
  }

  if (choices.length === 0) {
    const field = evaluateField(state);
    const winner = opponent(player);
    if (isKingUnprotected(player, state, field)) {
      return {
        ...state,
        status: winStatus(winner),
        selectedPieceId: null,
        history: [...state.history, snapshot(state)],
        message: `${playerName(player)} has no legal rescue`,
      };
    }
    return {
      ...state,
      status: winStatus(winner),
      selectedPieceId: null,
      history: [...state.history, snapshot(state)],
      message: `${playerName(player)} has no legal move`,
    };
  }

  const seed = options.seed ?? 0;
  const variety = Math.max(0, Math.min(options.variety ?? 0, 1));
  if (variety > 0) {
    const leader = choices[0].score;
    const loopBoost = Math.min(Math.max(0, currentBoardVisits - 1) * 0.18, 0.4);
    const exploration = Math.min(0.95, variety + loopBoost);
    const candidateWindow = choices
      .filter((choice) => leader - choice.score <= 120 + exploration * 320)
      .sort((left, right) => right.score - left.score);
    const shouldExplore = hashUnit(`${seed}:${state.turnNumber}:${player}:${stateKey(state)}:explore`) < exploration;
    if (shouldExplore && candidateWindow.length > 1) {
      const temperature = 70 + exploration * 360;
      const weights = candidateWindow.map((choice) => Math.exp((choice.score - leader) / temperature));
      const totalWeight = weights.reduce((total, weight) => total + weight, 0);
      let draw = hashUnit(`${seed}:${state.turnNumber}:${player}:${stateKey(state)}:draw`) * totalWeight;
      let selectedIndex = 0;
      for (let index = 0; index < weights.length; index += 1) {
        draw -= weights[index];
        if (draw <= 0) {
          selectedIndex = index;
          break;
        }
      }
      const [selected] = candidateWindow.splice(selectedIndex, 1);
      candidateWindow.unshift(selected);
    }
    choices.splice(0, candidateWindow.length, ...candidateWindow);
  }

  let fallback: GameState | null = null;
  const fullAnalysisChoices = choices.slice(0, nowMs() >= deadline ? 1 : fullAnalysisLimit);
  for (const choice of fullAnalysisChoices) {
    const result = applyMove(choice.pieceId, choice.destination, choice.tuned);
    if (!result.ok) continue;
    if (result.state.status === `${player}-won`) return { ...result.state, history: [...state.history, snapshot(state)] };
    fallback ??= result.state;
  }
  fallback ??= choices[0].preview;
  return fallback
    ? { ...fallback, history: [...state.history, snapshot(state)] }
    : { ...state, message: `${player === "red" ? "Red" : "Blue"} has no legal move` };
}
