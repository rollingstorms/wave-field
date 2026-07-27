import { evaluateField } from "../field/evaluateField";
import { isSquareCompatible } from "../field/projection";
import { COMPONENT_COUNTS } from "./constants";
import { snapshot } from "./initialState";
import { getLegalMoves } from "./movement";
import { applyMove } from "./rules";
import { activationOrderForProfile, isTuningWithinStrength } from "./tuning";
import type { Coefficient, GameState, PieceType, Player, PlayerComponents } from "./types";
import { getUnstablePieces, isKingUnprotected, markInstability } from "./victory";

const pieceTypes: PieceType[] = ["pawn", "rook", "spy", "king"];
const coefficients: Coefficient[] = [-1, 0, 1];
const materialValue = { pawn: 2, rook: 4, spy: 3, king: 100 } as const;
const exactCandidateLimit = 8;

function componentProfiles(pieceType: PieceType): Coefficient[][] {
  const count = COMPONENT_COUNTS[pieceType];
  const profiles: Coefficient[][] = [];

  function build(values: Coefficient[]) {
    if (values.length === count) {
      if (isTuningWithinStrength(pieceType, values)) {
        profiles.push(values);
      }
      return;
    }
    coefficients.forEach((value) => build([...values, value]));
  }

  build([]);
  return profiles;
}

function tuningCandidates(state: GameState, player: Player): PlayerComponents[] {
  const current = state.components[player];
  const candidates = [structuredClone(current)];
  const seen = new Set([JSON.stringify(current)]);

  for (const pieceType of pieceTypes) {
    for (const profile of componentProfiles(pieceType)) {
      const next = structuredClone(current);
      next[pieceType] = profile as never;
      const key = JSON.stringify(next);
      if (!seen.has(key)) {
        seen.add(key);
        candidates.push(next);
      }
    }
  }
  return candidates;
}

function scoreState(state: GameState, player: Player): number {
  if (state.status === `${player}-won`) return 1_000_000;
  if (state.status !== "playing") return -1_000_000;

  const enemy = player === "red" ? "blue" : "red";
  const field = evaluateField(state);
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

export function playHeuristicTurn(state: GameState, player: Player = "red"): GameState {
  if (state.status !== "playing" || state.currentPlayer !== player) return state;

  const choices: Array<{ tuned: GameState; pieceId: string; destination: { x: number; y: number }; score: number }> = [];
  let bestScore = Number.NEGATIVE_INFINITY;

  for (const profile of tuningCandidates(state, player)) {
    const components = structuredClone(state.components);
    components[player] = profile;
    const activationOrders = structuredClone(state.activationOrders);
    activationOrders[player] = activationOrderForProfile(profile);
    const tuned = markInstability(
      { ...state, components, activationOrders, selectedPieceId: null },
      evaluateField({ ...state, components, activationOrders }),
    );
    const field = evaluateField(tuned);

    for (const piece of tuned.pieces.filter((candidate) => candidate.owner === player)) {
      for (const destination of getLegalMoves(piece.id, tuned, field)) {
        const result = applyMove(piece.id, destination, tuned, { analyzeCheckmate: false });
        if (!result.ok) continue;
        const score = scoreState(result.state, player);
        if (choices.length < exactCandidateLimit || score > bestScore) {
          choices.push({ tuned, pieceId: piece.id, destination, score });
          choices.sort((a, b) => b.score - a.score);
          choices.length = Math.min(choices.length, exactCandidateLimit);
          bestScore = choices.at(-1)?.score ?? Number.NEGATIVE_INFINITY;
        }
      }
    }
  }

  if (choices.length === 0) return { ...state, message: `${player === "red" ? "Red" : "Blue"} has no legal move` };
  let fallback: GameState | null = null;
  for (const choice of choices) {
    const result = applyMove(choice.pieceId, choice.destination, choice.tuned);
    if (!result.ok) continue;
    if (result.state.status === `${player}-won`) return { ...result.state, history: [...state.history, snapshot(state)] };
    fallback ??= result.state;
  }
  return fallback
    ? { ...fallback, history: [...state.history, snapshot(state)] }
    : { ...state, message: `${player === "red" ? "Red" : "Blue"} has no legal move` };
}
