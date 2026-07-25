import { BOARD_SIZE } from "../game/constants";
import { projectFieldValue } from "../field/projection";
import { getLegalMoves, getPieceAt, samePosition } from "../game/movement";
import type { GameState, Position } from "../game/types";
import { Square } from "./Square";

interface BoardProps {
  state: GameState;
  field: number[][];
  highContrast: boolean;
  onSelect: (pieceId: string | null) => void;
  onMove: (pieceId: string, destination: Position) => void;
}

export function Board({ state, field, highContrast, onSelect, onMove }: BoardProps) {
  const selectedPiece = state.pieces.find((piece) => piece.id === state.selectedPieceId);
  const legalMoves = selectedPiece ? getLegalMoves(selectedPiece.id, state, field) : [];

  function handleSquare(position: Position) {
    const piece = getPieceAt(state, position);
    const isLegal = selectedPiece && legalMoves.some((move) => samePosition(move, position));
    if (selectedPiece && isLegal) {
      onMove(selectedPiece.id, position);
      return;
    }
    if (piece?.owner === state.currentPlayer && state.status === "playing") {
      onSelect(piece.id);
      return;
    }
    onSelect(null);
  }

  return (
    <section className="board-wrap" aria-label="Wave Field board">
      <div className="files top">{Array.from({ length: BOARD_SIZE }, (_, x) => <span key={x}>{x + 1}</span>)}</div>
      <div className="board-row-wrap">
        <div className="ranks left">{Array.from({ length: BOARD_SIZE }, (_, i) => <span key={i}>{BOARD_SIZE - i}</span>)}</div>
        <div className="board">
          {Array.from({ length: BOARD_SIZE }, (_, y) =>
            Array.from({ length: BOARD_SIZE }, (_, x) => {
              const position = { x, y };
              const piece = getPieceAt(state, position);
              return (
                <Square
                  key={`${x}-${y}`}
                  position={position}
                  territory={projectFieldValue(field[y][x])}
                  fieldValue={field[y][x]}
                  piece={piece}
                  legal={legalMoves.some((move) => samePosition(move, position))}
                  selected={piece?.id === selectedPiece?.id}
                  highContrast={highContrast}
                  onClick={() => handleSquare(position)}
                />
              );
            }),
          )}
        </div>
        <div className="ranks right">{Array.from({ length: BOARD_SIZE }, (_, i) => <span key={i}>{BOARD_SIZE - i}</span>)}</div>
      </div>
      <div className="files bottom">{Array.from({ length: BOARD_SIZE }, (_, x) => <span key={x}>{x + 1}</span>)}</div>
    </section>
  );
}
