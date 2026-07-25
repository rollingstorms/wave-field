import { BOARD_SIZE, PIECE_STRENGTH } from "../game/constants";
import type { ComponentDefinitions, GameState, Piece, Position } from "../game/types";
import { offset } from "./distance";
import { evaluateBasis } from "./kernels";

export function evaluatePieceContribution(
  piece: Piece,
  square: Position,
  state: GameState,
  definitions: ComponentDefinitions = state.definitions,
): number {
  const coefficients = state.components[piece.owner][piece.type];
  const bases = definitions[piece.type];
  const delta = offset(piece.position, square);
  return PIECE_STRENGTH[piece.type] * coefficients.reduce<number>((total, coefficient, index) => {
    return total + coefficient * evaluateBasis(bases[index], delta);
  }, 0);
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
  return Array.from({ length: BOARD_SIZE }, (_, y) =>
    Array.from({ length: BOARD_SIZE }, (_, x) =>
      state.pieces.reduce((total, piece) => total + evaluateSignedPieceContribution(piece, { x, y }, state, definitions), 0),
    ),
  );
}

export function contributionGrid(piece: Piece, state: GameState, definitions: ComponentDefinitions = state.definitions): number[][] {
  return Array.from({ length: BOARD_SIZE }, (_, y) =>
    Array.from({ length: BOARD_SIZE }, (_, x) => evaluateSignedPieceContribution(piece, { x, y }, state, definitions)),
  );
}
