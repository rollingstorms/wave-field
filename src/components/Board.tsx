import { useEffect, useMemo, useRef, useState } from "react";
import type {
  MouseEvent as ReactMouseEvent,
  PointerEvent as ReactPointerEvent,
  TouchEvent as ReactTouchEvent,
} from "react";
import { BOARD_SIZE } from "../game/constants";
import { contributionGrid, evaluateField, evaluateTypeFields } from "../field/evaluateField";
import type { TypeFields } from "../field/evaluateField";
import { projectFieldValue } from "../field/projection";
import { getPieceAt, samePosition } from "../game/movement";
import { applyMove, getPlayableMoves } from "../game/rules";
import type { GameState, Position } from "../game/types";
import { markInstability } from "../game/victory";
import { Square } from "./Square";

interface BoardProps {
  state: GameState;
  field: number[][];
  typeFields: TypeFields;
  highContrast: boolean;
  showTypeSums: boolean;
  locked?: boolean;
  onSelect: (pieceId: string | null) => void;
  onMove: (pieceId: string, destination: Position) => void;
}

interface ActiveDrag {
  pieceId: string;
  contactId: number;
  input: "pointer" | "touch";
  start: Position;
  legalMoves: Position[];
}

interface LossPop {
  id: string;
  position: Position;
}

export function Board({ state, field, typeFields, highContrast, showTypeSums, locked = false, onSelect, onMove }: BoardProps) {
  const boardRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<ActiveDrag | null>(null);
  const previousPiecesRef = useRef(state.pieces);
  const lossPopTimersRef = useRef<Array<ReturnType<typeof globalThis.setTimeout>>>([]);
  const suppressClickRef = useRef(false);
  const [draggingPieceId, setDraggingPieceId] = useState<string | null>(null);
  const [dragPreview, setDragPreview] = useState<Position | null>(null);
  const [lossPops, setLossPops] = useState<LossPop[]>([]);
  const selectedPiece = state.pieces.find((piece) => piece.id === state.selectedPieceId);
  const interactionPiece = state.pieces.find((piece) => piece.id === draggingPieceId) ?? selectedPiece;
  const legalMoves = !locked && interactionPiece ? getPlayableMoves(interactionPiece.id, state, field) : [];
  const previewState = useMemo<GameState>(() => {
    if (!draggingPieceId || !dragPreview) return state;
    const moved = {
      ...state,
      pieces: state.pieces.map((piece) =>
        piece.id === draggingPieceId ? { ...piece, position: dragPreview } : piece,
      ),
    };
    return markInstability(moved, evaluateField(moved));
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
  const lossPopKeys = useMemo(
    () => new Set(lossPops.map((pop) => `${pop.position.x}:${pop.position.y}`)),
    [lossPops],
  );
  const riskyMoveKeys = useMemo(() => {
    if (!selectedPiece) return new Set<string>();
    const ownPieceIds = new Set(state.pieces.filter((piece) => piece.owner === state.currentPlayer).map((piece) => piece.id));
    return new Set(legalMoves.flatMap((move) => {
      const result = applyMove(selectedPiece.id, move, state);
      if (!result.ok) return [];
      const remainingIds = new Set(result.state.pieces.map((piece) => piece.id));
      return [...ownPieceIds].some((id) => !remainingIds.has(id)) ? [`${move.x}:${move.y}`] : [];
    }));
  }, [legalMoves, selectedPiece, state]);

  useEffect(() => {
    const currentIds = new Set(state.pieces.map((piece) => piece.id));
    const lost = previousPiecesRef.current.filter((piece) => !currentIds.has(piece.id));
    previousPiecesRef.current = state.pieces;
    if (lost.length === 0) return;

    const created = lost.map((piece) => ({
      id: `${piece.id}:${globalThis.performance.now()}`,
      position: piece.position,
    }));
    setLossPops((pops) => [...pops, ...created]);
    const timer = globalThis.setTimeout(() => {
      setLossPops((pops) => pops.filter((pop) => !created.some((candidate) => candidate.id === pop.id)));
      lossPopTimersRef.current = lossPopTimersRef.current.filter((candidate) => candidate !== timer);
    }, 560);
    lossPopTimersRef.current.push(timer);
  }, [state.pieces]);

  useEffect(() => {
    return () => {
      for (const timer of lossPopTimersRef.current) globalThis.clearTimeout(timer);
    };
  }, []);

  function positionFromPointer(clientX: number, clientY: number): Position | null {
    const board = boardRef.current;
    if (!board) return null;
    const bounds = board.getBoundingClientRect();
    const x = Math.floor(((clientX - bounds.left) / bounds.width) * BOARD_SIZE);
    const y = Math.floor(((clientY - bounds.top) / bounds.height) * BOARD_SIZE);
    const position = { x, y };
    return x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE ? position : null;
  }

  function startDrag(contactId: number, input: ActiveDrag["input"], clientX: number, clientY: number) {
    if (locked || state.status !== "playing" || dragRef.current) return false;
    const position = positionFromPointer(clientX, clientY);
    const piece = position ? getPieceAt(state, position) : undefined;
    if (!piece || piece.owner !== state.currentPlayer) return false;

    const moves = getPlayableMoves(piece.id, state, field);
    dragRef.current = { pieceId: piece.id, contactId, input, start: piece.position, legalMoves: moves };
    setDraggingPieceId(piece.id);
    setDragPreview(piece.position);
    onSelect(piece.id);
    return true;
  }

  function updateDrag(contactId: number, input: ActiveDrag["input"], clientX: number, clientY: number) {
    const drag = dragRef.current;
    if (!drag || drag.contactId !== contactId || drag.input !== input) return false;
    const position = positionFromPointer(clientX, clientY);
    const legal = position && drag.legalMoves.some((move) => samePosition(move, position));
    setDragPreview(legal ? position : drag.start);
    return true;
  }

  function completeDrag(contactId: number, input: ActiveDrag["input"], clientX: number, clientY: number, commit: boolean) {
    const drag = dragRef.current;
    if (!drag || drag.contactId !== contactId || drag.input !== input) return false;
    const position = positionFromPointer(clientX, clientY);
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

    dragRef.current = null;
    setDraggingPieceId(null);
    setDragPreview(null);
    return true;
  }

  function handlePointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    if (event.pointerType === "touch" || event.button !== 0) return;
    if (!startDrag(event.pointerId, "pointer", event.clientX, event.clientY)) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
  }

  function handlePointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    if (updateDrag(event.pointerId, "pointer", event.clientX, event.clientY)) event.preventDefault();
  }

  function finishPointerDrag(event: ReactPointerEvent<HTMLDivElement>, commit: boolean) {
    if (!completeDrag(event.pointerId, "pointer", event.clientX, event.clientY, commit)) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    event.preventDefault();
  }

  function handleTouchStart(event: ReactTouchEvent<HTMLDivElement>) {
    const touch = event.changedTouches[0];
    if (!touch || !startDrag(touch.identifier, "touch", touch.clientX, touch.clientY)) return;
    event.preventDefault();
  }

  function handleTouchMove(event: ReactTouchEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (!drag || drag.input !== "touch") return;
    const touch = Array.from(event.touches).find((candidate) => candidate.identifier === drag.contactId);
    if (touch && updateDrag(touch.identifier, "touch", touch.clientX, touch.clientY)) event.preventDefault();
  }

  function finishTouchDrag(event: ReactTouchEvent<HTMLDivElement>, commit: boolean) {
    const drag = dragRef.current;
    if (!drag || drag.input !== "touch") return;
    const touch = Array.from(event.changedTouches).find((candidate) => candidate.identifier === drag.contactId);
    const clientX = touch?.clientX ?? -1;
    const clientY = touch?.clientY ?? -1;
    if (completeDrag(drag.contactId, "touch", clientX, clientY, commit)) event.preventDefault();
  }

  function handleClickCapture(event: ReactMouseEvent<HTMLDivElement>) {
    if (!suppressClickRef.current) return;
    suppressClickRef.current = false;
    event.preventDefault();
    event.stopPropagation();
  }

  function handleSquare(position: Position) {
    if (locked) return;
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
          onPointerUp={(event) => finishPointerDrag(event, true)}
          onPointerCancel={(event) => finishPointerDrag(event, false)}
          onTouchStart={handleTouchStart}
          onTouchMove={handleTouchMove}
          onTouchEnd={(event) => finishTouchDrag(event, true)}
          onTouchCancel={(event) => finishTouchDrag(event, false)}
          onClickCapture={handleClickCapture}
        >
          {Array.from({ length: BOARD_SIZE }, (_, y) =>
            Array.from({ length: BOARD_SIZE }, (_, x) => {
              const position = { x, y };
              const piece = getPieceAt(previewState, position);
              const influenceValue = influenceGrid?.[y][x] ?? 0;
              const influence = Math.abs(influenceValue);
              return (
                <Square
                  key={`${x}-${y}`}
                  position={position}
                  territory={projectFieldValue(displayField[y][x])}
                  fieldValue={displayField[y][x]}
                  piece={piece}
                  legal={legalMoves.some((move) => samePosition(move, position))}
                  risky={riskyMoveKeys.has(`${x}:${y}`)}
                  selected={Boolean(piece && interactionPiece && piece.id === interactionPiece.id)}
                  dragging={piece?.id === draggingPieceId}
                  dragPreview={Boolean(draggingPieceId && dragPreview && samePosition(dragPreview, position))}
                  influenceTerritory={influenceGrid ? projectFieldValue(influenceValue) : null}
                  influenceOpacity={maximumInfluence > 0 && influence > 0
                    ? 0.45 + (influence / maximumInfluence) * 0.55
                    : 0}
                  highContrast={highContrast}
                  typeSums={showTypeSums ? {
                    pawn: displayTypeFields.pawn[y][x],
                    rook: displayTypeFields.rook[y][x],
                    spy: displayTypeFields.spy[y][x],
                    king: displayTypeFields.king[y][x],
                  } : null}
                  lossPop={lossPopKeys.has(`${x}:${y}`)}
                  onClick={() => handleSquare(position)}
                />
              );
            }),
          )}
        </div>
        <div className="ranks right">{Array.from({ length: BOARD_SIZE }, (_, i) => <span key={i}>{BOARD_SIZE - i}</span>)}</div>
      </div>
      <div className="files bottom">{Array.from({ length: BOARD_SIZE }, (_, x) => <span key={x}>{x + 1}</span>)}</div>
      {selectedPiece?.unstable && (
        <p className="piece-alert-hint" role="status" aria-live="polite">
          <strong>{selectedPiece.type === "king" ? "UNPROTECTED KING" : "UNSTABLE PIECE"}</strong>
          {selectedPiece.type === "king"
            ? "Tune the field until the king's square is friendly or neutral before moving. You cannot end the turn with an unprotected king."
            : "Move this piece or tune the field until its square is friendly or neutral. Otherwise it disappears when the turn ends."}
        </p>
      )}
    </section>
  );
}
