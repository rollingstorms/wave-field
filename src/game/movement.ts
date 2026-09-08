import { BOARD_SIZE } from "./constants";
import type { GameState, Piece, Position, PrecisePosition } from "./types";
import { evaluateContinuousFieldValue, evaluateField } from "../field/evaluateField";
import { isSquareCompatible } from "../field/projection";
import { rustLegalMoves } from "./rustEngine";

export const CONTINUOUS_MOVEMENT_SAMPLES_PER_SQUARE = 9;
export const CONTINUOUS_PIECE_RADIUS = 0.35;
const PRECISE_EPSILON = 1e-9;

export function inBounds(position: Position): boolean {
  return position.x >= 0 && position.x < BOARD_SIZE && position.y >= 0 && position.y < BOARD_SIZE;
}

export function samePosition(a: Position, b: Position): boolean {
  return a.x === b.x && a.y === b.y;
}

export function samePrecisePosition(a: PrecisePosition, b: PrecisePosition, epsilon = PRECISE_EPSILON): boolean {
  return Math.abs(a.x - b.x) <= epsilon && Math.abs(a.y - b.y) <= epsilon;
}

export function getPieceAt(state: GameState, position: Position): Piece | undefined {
  return state.pieces.find((piece) => samePosition(piece.position, position));
}

export function getPieceAtPrecise(state: GameState, position: PrecisePosition): Piece | undefined {
  return state.pieces.find((piece) => samePrecisePosition(piece.position, position));
}

function isSpectrallyPassable(piece: Piece, position: Position, field: number[][]): boolean {
  return piece.type === "spy" || isSquareCompatible(piece.owner, field[position.y][position.x]);
}

function movementStep(origin: Position, destination: Position): Position | null {
  const dx = destination.x - origin.x;
  const dy = destination.y - origin.y;
  if (dx === 0 && dy === 0) return null;
  if (dx !== 0 && dy !== 0 && Math.abs(dx) !== Math.abs(dy)) return null;
  return { x: Math.sign(dx), y: Math.sign(dy) };
}

export function canPieceEnter(piece: Piece, destination: Position, state: GameState, field: number[][]): boolean {
  if (!inBounds(destination)) return false;
  const step = movementStep(piece.position, destination);
  if (!step) return false;

  let position = { x: piece.position.x + step.x, y: piece.position.y + step.y };
  while (inBounds(position)) {
    if (getPieceAt(state, position) || !isSpectrallyPassable(piece, position, field)) return false;
    if (samePosition(position, destination)) return true;
    position = { x: position.x + step.x, y: position.y + step.y };
  }
  return false;
}

export function inContinuousBounds(position: PrecisePosition): boolean {
  return position.x >= -PRECISE_EPSILON
    && position.x <= BOARD_SIZE - 1 + PRECISE_EPSILON
    && position.y >= -PRECISE_EPSILON
    && position.y <= BOARD_SIZE - 1 + PRECISE_EPSILON;
}

function continuousBoardPoints(samplesPerSquare: number): PrecisePosition[] {
  const steps = (BOARD_SIZE - 1) * samplesPerSquare;
  return Array.from({ length: steps + 1 }, (_, yStep) =>
    Array.from({ length: steps + 1 }, (_, xStep) => ({
      x: xStep / samplesPerSquare,
      y: yStep / samplesPerSquare,
    })),
  ).flat();
}

function segmentInteriorSamples(origin: PrecisePosition, destination: PrecisePosition, samplesPerSquare: number): PrecisePosition[] {
  const dx = destination.x - origin.x;
  const dy = destination.y - origin.y;
  const distance = Math.hypot(dx, dy);
  const steps = Math.max(1, Math.ceil(distance * samplesPerSquare));
  return Array.from({ length: steps }, (_, index) => {
    const amount = (index + 1) / steps;
    return {
      x: origin.x + dx * amount,
      y: origin.y + dy * amount,
    };
  });
}

function squaredDistance(left: PrecisePosition, right: PrecisePosition): number {
  return (left.x - right.x) ** 2 + (left.y - right.y) ** 2;
}

function pointToSegmentDistance(point: PrecisePosition, start: PrecisePosition, end: PrecisePosition): number {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared <= PRECISE_EPSILON) return Math.hypot(point.x - start.x, point.y - start.y);
  const rawAmount = ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared;
  const amount = Math.max(0, Math.min(1, rawAmount));
  return Math.hypot(point.x - (start.x + dx * amount), point.y - (start.y + dy * amount));
}

function continuousPieceBlocksPath(piece: Piece, destination: PrecisePosition, state: GameState): boolean {
  const minimumCenterDistance = CONTINUOUS_PIECE_RADIUS * 2;
  const minimumCenterDistanceSquared = minimumCenterDistance ** 2;
  return state.pieces.some((other) => {
    if (other.id === piece.id) return false;
    if (squaredDistance(other.position, destination) < minimumCenterDistanceSquared - PRECISE_EPSILON) return true;
    return pointToSegmentDistance(other.position, piece.position, destination) < minimumCenterDistance - PRECISE_EPSILON;
  });
}

function isContinuouslyPassable(
  piece: Piece,
  position: PrecisePosition,
  state: GameState,
  fieldValueAt: (point: PrecisePosition) => number,
): boolean {
  return piece.type === "spy" || isSquareCompatible(piece.owner, fieldValueAt(position));
}

export interface ContinuousMovementOptions {
  samplesPerSquare?: number;
  fieldValueAt?: (point: PrecisePosition) => number;
}

export function canContinuousPieceEnter(
  piece: Piece,
  destination: PrecisePosition,
  state: GameState,
  options: ContinuousMovementOptions = {},
): boolean {
  if (!inContinuousBounds(destination) || samePrecisePosition(piece.position, destination)) return false;
  if (continuousPieceBlocksPath(piece, destination, state)) return false;
  const samplesPerSquare = options.samplesPerSquare ?? CONTINUOUS_MOVEMENT_SAMPLES_PER_SQUARE;
  const fieldValueAt = options.fieldValueAt ?? ((point) => evaluateContinuousFieldValue(state, point));
  for (const position of segmentInteriorSamples(piece.position, destination, samplesPerSquare)) {
    if (!isContinuouslyPassable(piece, position, state, fieldValueAt)) return false;
  }
  return true;
}

export function getContinuousLegalMoves(
  pieceId: string,
  state: GameState,
  options: ContinuousMovementOptions = {},
): PrecisePosition[] {
  const piece = state.pieces.find((candidate) => candidate.id === pieceId);
  if (!piece) return [];
  const samplesPerSquare = options.samplesPerSquare ?? CONTINUOUS_MOVEMENT_SAMPLES_PER_SQUARE;
  const fieldValueAt = options.fieldValueAt ?? ((point) => evaluateContinuousFieldValue(state, point));
  return continuousBoardPoints(samplesPerSquare).filter((destination) =>
    canContinuousPieceEnter(piece, destination, state, { samplesPerSquare, fieldValueAt }));
}

export function getLegalMoves(pieceId: string, state: GameState, field?: number[][]): Position[] {
  const rustMoves = field ? null : rustLegalMoves(pieceId, state);
  if (rustMoves) return rustMoves;
  const activeField = field ?? evaluateField(state);
  const piece = state.pieces.find((candidate) => candidate.id === pieceId);
  if (!piece) return [];
  const moves: Position[] = [];
  for (let dy = -1; dy <= 1; dy += 1) {
    for (let dx = -1; dx <= 1; dx += 1) {
      if (dx === 0 && dy === 0) continue;
      let destination = { x: piece.position.x + dx, y: piece.position.y + dy };
      while (inBounds(destination)) {
        if (getPieceAt(state, destination) || !isSpectrallyPassable(piece, destination, activeField)) break;
        moves.push(destination);
        destination = { x: destination.x + dx, y: destination.y + dy };
      }
    }
  }
  return moves;
}
