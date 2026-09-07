import { BOARD_SIZE, PIECE_STRENGTH } from "../game/constants";
import type { ComponentDefinitions, GameState, Piece, PieceType, Position, PrecisePosition } from "../game/types";
import { rustEvaluateField } from "../game/rustEngine";
import { isAmpSquare } from "../game/variants";
import { offset } from "./distance";
import { evaluateComponentBasis } from "./kernels";

const AMP_MULTIPLIER = 2;

function lerp(start: number, end: number, amount: number): number {
  return start + (end - start) * amount;
}

function evaluatePieceContributionAtOffset(
  piece: Piece,
  delta: Position,
  state: GameState,
  definitions: ComponentDefinitions,
): number {
  const coefficients = state.components[piece.owner][piece.type];
  const bases = definitions[piece.type];
  const multiplier = isAmpSquare(piece.position, state.ampSquares) ? AMP_MULTIPLIER : 1;
  if (delta.x === 0 && delta.y === 0) return state.homeEnergy[piece.type] * multiplier;
  const raw = coefficients.reduce(
    (totals, coefficient, index) => {
      const value = coefficient * evaluateComponentBasis(piece.type, bases[index], delta);
      if (value > 0) totals.positive += value;
      else if (value < 0) totals.negative += value;
      return totals;
    },
    { positive: 0, negative: 0 },
  );
  const scale = state.waveScales[piece.type];
  return multiplier * PIECE_STRENGTH[piece.type] * (raw.positive * scale.friendly + raw.negative * scale.hostile);
}

export function evaluatePieceContribution(
  piece: Piece,
  square: Position,
  state: GameState,
  definitions: ComponentDefinitions = state.definitions,
): number {
  const delta = offset(piece.position, square);
  return evaluatePieceContributionAtOffset(piece, delta, state, definitions);
}

export function evaluateSignedPieceContribution(
  piece: Piece,
  square: Position,
  state: GameState,
  definitions: ComponentDefinitions = state.definitions,
): number {
  const sigma = piece.owner === "red" ? 1 : -1;
  return sigma * evaluatePieceContribution(piece, square, state, definitions);
}

export function evaluateContinuousPieceContribution(
  piece: Piece,
  point: PrecisePosition,
  state: GameState,
  definitions: ComponentDefinitions = state.definitions,
): number {
  const delta = {
    x: point.x - piece.position.x,
    y: point.y - piece.position.y,
  };
  const x0 = Math.floor(delta.x);
  const y0 = Math.floor(delta.y);
  const x1 = x0 + 1;
  const y1 = y0 + 1;
  const tx = delta.x - x0;
  const ty = delta.y - y0;
  const bottomLeft = evaluatePieceContributionAtOffset(piece, { x: x0, y: y0 }, state, definitions);
  if (tx === 0 && ty === 0) return bottomLeft;
  const bottomRight = evaluatePieceContributionAtOffset(piece, { x: x1, y: y0 }, state, definitions);
  const topLeft = evaluatePieceContributionAtOffset(piece, { x: x0, y: y1 }, state, definitions);
  const topRight = evaluatePieceContributionAtOffset(piece, { x: x1, y: y1 }, state, definitions);
  return lerp(lerp(bottomLeft, bottomRight, tx), lerp(topLeft, topRight, tx), ty);
}

export function evaluateSignedContinuousPieceContribution(
  piece: Piece,
  point: PrecisePosition,
  state: GameState,
  definitions: ComponentDefinitions = state.definitions,
): number {
  const sigma = piece.owner === "red" ? 1 : -1;
  return sigma * evaluateContinuousPieceContribution(piece, point, state, definitions);
}

export function evaluateInstantField(state: GameState, definitions: ComponentDefinitions = state.definitions): number[][] {
  if (definitions === state.definitions) {
    const rustField = rustEvaluateField(state);
    if (rustField) return rustField;
  }
  return Array.from({ length: BOARD_SIZE }, (_, y) =>
    Array.from({ length: BOARD_SIZE }, (_, x) =>
      state.variant === "continuous"
        ? evaluateContinuousFieldValue(state, { x, y }, definitions)
        : state.pieces.reduce((total, piece) => total + evaluateSignedPieceContribution(piece, { x, y }, state, definitions), 0),
    ),
  );
}

export interface ContinuousFieldSample {
  value: number;
  point: PrecisePosition;
}

export function evaluateContinuousField(
  state: GameState,
  samplesPerSquare = 8,
  definitions: ComponentDefinitions = state.definitions,
): ContinuousFieldSample[][] {
  const resolution = BOARD_SIZE * samplesPerSquare;
  return Array.from({ length: resolution }, (_, row) =>
    Array.from({ length: resolution }, (_, x) => {
      const point = {
        x: (x + 0.5) / samplesPerSquare - 0.5,
        y: BOARD_SIZE - 0.5 - (row + 0.5) / samplesPerSquare,
      };
      return {
        point,
        value: state.pieces.reduce(
          (total, piece) => total + evaluateSignedContinuousPieceContribution(piece, point, state, definitions),
          0,
        ),
      };
    }),
  );
}

export function evaluateContinuousFieldValue(
  state: GameState,
  point: PrecisePosition,
  definitions: ComponentDefinitions = state.definitions,
): number {
  return state.pieces.reduce(
    (total, piece) => total + evaluateSignedContinuousPieceContribution(piece, point, state, definitions),
    0,
  );
}

export function addFields(left: number[][], right: number[][]): number[][] {
  return left.map((row, y) => row.map((value, x) => value + (right[y]?.[x] ?? 0)));
}

export function evaluateField(state: GameState, definitions: ComponentDefinitions = state.definitions): number[][] {
  if (definitions === state.definitions && state.entropyField) return state.entropyField;
  return evaluateInstantField(state, definitions);
}

export type TypeFields = Record<PieceType, number[][]>;

export function evaluateTypeFields(
  state: GameState,
  definitions: ComponentDefinitions = state.definitions,
): TypeFields {
  const pieceTypes: PieceType[] = ["pawn", "rook", "spy", "king"];
  return Object.fromEntries(pieceTypes.map((pieceType) => {
    const pieces = state.pieces.filter((piece) => piece.type === pieceType);
    const field = Array.from({ length: BOARD_SIZE }, (_, y) =>
      Array.from({ length: BOARD_SIZE }, (_, x) =>
        pieces.reduce(
            (total, piece) => total + (
              state.variant === "continuous"
                ? evaluateSignedContinuousPieceContribution(piece, { x, y }, state, definitions)
                : evaluateSignedPieceContribution(piece, { x, y }, state, definitions)
            ),
            0,
        ),
      ),
    );
    return [pieceType, field];
  })) as TypeFields;
}

export function contributionGrid(piece: Piece, state: GameState, definitions: ComponentDefinitions = state.definitions): number[][] {
  return Array.from({ length: BOARD_SIZE }, (_, y) =>
    Array.from({ length: BOARD_SIZE }, (_, x) => (
      state.variant === "continuous"
        ? evaluateSignedContinuousPieceContribution(piece, { x, y }, state, definitions)
        : evaluateSignedPieceContribution(piece, { x, y }, state, definitions)
    )),
  );
}
