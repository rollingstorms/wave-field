import { BOARD_SIZE, PIECE_STRENGTH } from "../game/constants";
import type { ComponentDefinitions, GameState, Piece, PieceType, Position } from "../game/types";
import { rustEvaluateField } from "../game/rustEngine";
import { offset } from "./distance";
import { evaluateComponentBasis } from "./kernels";

export function evaluatePieceContribution(
  piece: Piece,
  square: Position,
  state: GameState,
  definitions: ComponentDefinitions = state.definitions,
): number {
  const coefficients = state.components[piece.owner][piece.type];
  const bases = definitions[piece.type];
  const delta = offset(piece.position, square);
  if (delta.x === 0 && delta.y === 0) return state.homeEnergy[piece.type];
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
  return PIECE_STRENGTH[piece.type] * (raw.positive * scale.friendly + raw.negative * scale.hostile);
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

export function evaluateField(state: GameState, definitions: ComponentDefinitions = state.definitions): number[][] {
  if (definitions === state.definitions) {
    const rustField = rustEvaluateField(state);
    if (rustField) return rustField;
  }
  return Array.from({ length: BOARD_SIZE }, (_, y) =>
    Array.from({ length: BOARD_SIZE }, (_, x) =>
      state.pieces.reduce((total, piece) => total + evaluateSignedPieceContribution(piece, { x, y }, state, definitions), 0),
    ),
  );
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
            (total, piece) => total + evaluateSignedPieceContribution(piece, { x, y }, state, definitions),
            0,
        ),
      ),
    );
    return [pieceType, field];
  })) as TypeFields;
}

export function contributionGrid(piece: Piece, state: GameState, definitions: ComponentDefinitions = state.definitions): number[][] {
  return Array.from({ length: BOARD_SIZE }, (_, y) =>
    Array.from({ length: BOARD_SIZE }, (_, x) => evaluateSignedPieceContribution(piece, { x, y }, state, definitions)),
  );
}
