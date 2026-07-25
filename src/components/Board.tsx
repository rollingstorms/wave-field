import { useMemo, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent, PointerEvent as ReactPointerEvent } from "react";
import { BOARD_SIZE } from "../game/constants";
import { contributionGrid, evaluateField, evaluateTypeFields } from "../field/evaluateField";
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

interface ActiveDrag {
  pieceId: string;
  pointerId: number;
  start: Position;
  legalMoves: Position[];
}

export function Board({ state, field, typeFields, highContrast, showTypeSums, onSelect, onMove }: BoardProps) {
  const boardRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<ActiveDrag | null>(null);
  const suppressClickRef = useRef(false);
  const [draggingPieceId, setDraggingPieceId] = useState<string | null>(null);
  const [dragPreview, setDragPreview] = useState<Position | null>(null);
  const selectedPiece = state.pieces.find((piece) => piece.id === state.selectedPieceId);
  const interactionPiece = state.pieces.find((piece) => piece.id === draggingPieceId) ?? selectedPiece;
  const legalMoves = interactionPiece ? getLegalMoves(interactionPiece.id, state, field) : [];
  const previewState = useMemo<GameState>(() => {
    if (!draggingPieceId || !dragPreview) return state;
    return {
      ...state,
      pieces: state.pieces.map((piece) =>
        piece.id === draggingPieceId ? { ...piece, position: dragPreview } : piece,
      ),
    };
  }, [dragPreview, draggingPieceId, state]);
  const previewing = Boolean(draggingPieceId && dragPreview);
  const displayField = useMemo(
    () => previewing ? evaluateField(previewState) : field,
    [field, previewState, previewing],
  );
  const displayTypeFields = useMemo(
    () => previewing ? evaluateTypeFields(previewState) : typeFields,
    [previewState, previewing, typeFields],
  );
  const displaySelectedPiece = interactionPiece
    ? previewState.pieces.find((piece) => piece.id === interactionPiece.id)
    : undefined;
  const influenceGrid = useMemo(
    () => displaySelectedPiece ? contributionGrid(displaySelectedPiece, previewState) : null,
    [displaySelectedPiece, previewState],
  );
  const maximumInfluence = influenceGrid
    ? Math.max(...influenceGrid.flat().map((value) => Math.abs(value)), 0)
    : 0;

  function positionFromPointer(clientX: number, clientY: number): Position | null {
    const board = boardRef.current;
    if (!board) return null;
    const bounds = board.getBoundingClientRect();
    const x = Math.floor(((clientX - bounds.left) / bounds.width) * BOARD_SIZE);
    const y = Math.floor(((clientY - bounds.top) / bounds.height) * BOARD_SIZE);
    const position = { x, y };
    return x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE ? position : null;
  }

  function handlePointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    if (event.button !== 0 || state.status !== "playing") return;
    const position = positionFromPointer(event.clientX, event.clientY);
    const piece = position ? getPieceAt(state, position) : undefined;
    if (!piece || piece.owner !== state.currentPlayer) return;

    const moves = getLegalMoves(piece.id, state, field);
    dragRef.current = { pieceId: piece.id, pointerId: event.pointerId, start: piece.position, legalMoves: moves };
    setDraggingPieceId(piece.id);
    setDragPreview(piece.position);
    onSelect(piece.id);
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
  }

  function handlePointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const position = positionFromPointer(event.clientX, event.clientY);
    const legal = position && drag.legalMoves.some((move) => samePosition(move, position));
    setDragPreview(legal ? position : drag.start);
    event.preventDefault();
  }

  function finishDrag(event: ReactPointerEvent<HTMLDivElement>, commit: boolean) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const position = positionFromPointer(event.clientX, event.clientY);
    const destination = commit && position && drag.legalMoves.some((move) => samePosition(move, position))
      ? position
      : null;

    if (destination && !samePosition(destination, drag.start)) {
      suppressClickRef.current = true;
      globalThis.setTimeout(() => {
        suppressClickRef.current = false;
      }, 0);
      onMove(drag.pieceId, destination);
    } else {
      onSelect(drag.pieceId);
    }

    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    dragRef.current = null;
    setDraggingPieceId(null);
    setDragPreview(null);
    event.preventDefault();
  }

  function handleClickCapture(event: ReactMouseEvent<HTMLDivElement>) {
    if (!suppressClickRef.current) return;
    suppressClickRef.current = false;
    event.preventDefault();
    event.stopPropagation();
  }

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
        <div
          className={`board ${draggingPieceId ? "dragging" : ""}`}
          ref={boardRef}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={(event) => finishDrag(event, true)}
          onPointerCancel={(event) => finishDrag(event, false)}
          onClickCapture={handleClickCapture}
        >
          {Array.from({ length: BOARD_SIZE }, (_, y) =>
            Array.from({ length: BOARD_SIZE }, (_, x) => {
              const position = { x, y };
              const piece = getPieceAt(previewState, position);
              const influence = influenceGrid ? Math.abs(influenceGrid[y][x]) : 0;
              return (
                <Square
                  key={`${x}-${y}`}
                  position={position}
                  territory={projectFieldValue(displayField[y][x])}
                  fieldValue={displayField[y][x]}
                  piece={piece}
                  legal={legalMoves.some((move) => samePosition(move, position))}
                  selected={Boolean(piece && interactionPiece && piece.id === interactionPiece.id)}
                  dragging={piece?.id === draggingPieceId}
                  dragPreview={Boolean(draggingPieceId && dragPreview && samePosition(dragPreview, position))}
                  influenceOpacity={maximumInfluence > 0 && influence > 0
                    ? 0.22 + (influence / maximumInfluence) * 0.78
                    : 0}
                  highContrast={highContrast}
                  typeSums={showTypeSums ? {
                    pawn: displayTypeFields.pawn[y][x],
                    rook: displayTypeFields.rook[y][x],
                    spy: displayTypeFields.spy[y][x],
                    king: displayTypeFields.king[y][x],
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
