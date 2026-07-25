import { BOARD_SIZE } from "../game/constants";
import type { TypeFields } from "../field/evaluateField";
import { projectFieldValue } from "../field/projection";
import { getLegalMoves, getPieceAt, samePosition } from "../game/movement";
import type { GameState, Position } from "../game/types";
import { Square } from "./Square";

interface BoardProps {
  state: GameState;
  field: number[][];
  typeFields: TypeFields;
  highContrast: boolean;
  showTypeSums: boolean;
  onSelect: (pieceId: string | null) => void;
  onMove: (pieceId: string, destination: Position) => void;
}

export function Board({ state, field, typeFields, highContrast, showTypeSums, onSelect, onMove }: BoardProps) {
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
      {showTypeSums && (
        <div className="type-sum-key" aria-label="Type sum corner key">
          <strong>TYPE SUMS</strong>
          <span>P ↖</span>
          <span>R ↗</span>
          <span>S ↙</span>
          <span>K ↘</span>
        </div>
      )}
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
                  typeSums={showTypeSums ? {
                    pawn: typeFields.pawn[y][x],
                    rook: typeFields.rook[y][x],
                    spy: typeFields.spy[y][x],
                    king: typeFields.king[y][x],
                  } : null}
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
