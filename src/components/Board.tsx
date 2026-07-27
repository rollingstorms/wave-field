import { useEffect, useMemo, useRef, useState } from "react";
import { Flag, Lightbulb } from "lucide-react";
import type {
  MouseEvent as ReactMouseEvent,
  PointerEvent as ReactPointerEvent,
  TouchEvent as ReactTouchEvent,
} from "react";
import { BOARD_SIZE } from "../game/constants";
import { createCmykEnergyGrid, ENERGY_CHANNELS } from "../field/cmykEnergy";
import type { EnergyChannelState } from "../field/cmykEnergy";
import { contributionGrid, evaluateField, evaluateTypeFields } from "../field/evaluateField";
import type { TypeFields } from "../field/evaluateField";
import { projectFieldValue } from "../field/projection";
import { getLegalMoves, getPieceAt, samePosition } from "../game/movement";
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
  energyView: boolean;
  energyChannels: EnergyChannelState;
  locked?: boolean;
  onSelect: (pieceId: string | null) => void;
  onMove: (pieceId: string, destination: Position) => void;
  onResign: () => void;
  onHint: () => void;
  hintSearching?: boolean;
  onToggleEnergyChannel: (pieceType: keyof EnergyChannelState) => void;
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

export function Board({ state, field, typeFields, highContrast, showTypeSums, energyView, energyChannels, locked = false, onSelect, onMove, onResign, onHint, hintSearching = false, onToggleEnergyChannel }: BoardProps) {
  const boardRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<ActiveDrag | null>(null);
  const previousPiecesRef = useRef(state.pieces);
  const lossPopTimersRef = useRef<Array<ReturnType<typeof globalThis.setTimeout>>>([]);
  const suppressClickRef = useRef(false);
  const [draggingPieceId, setDraggingPieceId] = useState<string | null>(null);
  const [dragPreview, setDragPreview] = useState<Position | null>(null);
  const [lossPops, setLossPops] = useState<LossPop[]>([]);
  const [energySelection, setEnergySelection] = useState<Position | null>(null);
  const selectedPiece = state.pieces.find((piece) => piece.id === state.selectedPieceId);
  const interactionPiece = energyView ? undefined : state.pieces.find((piece) => piece.id === draggingPieceId) ?? selectedPiece;
  const reachableMoves = !locked && interactionPiece ? getLegalMoves(interactionPiece.id, state, field) : [];
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
  const energyGrid = useMemo(
    () => createCmykEnergyGrid(displayTypeFields, energyChannels),
    [displayTypeFields, energyChannels],
  );
  const selectedEnergy = energySelection ? energyGrid[energySelection.y][energySelection.x] : null;
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
  const playableMoveKeys = useMemo(
    () => new Set(legalMoves.map((move) => `${move.x}:${move.y}`)),
    [legalMoves],
  );
  const kingBlockedMoveKeys = useMemo(() => {
    if (!interactionPiece) return new Set<string>();
    return new Set(reachableMoves.flatMap((move) => {
      const key = `${move.x}:${move.y}`;
      if (playableMoveKeys.has(key)) return [];
      const result = applyMove(interactionPiece.id, move, state);
      return result.reason?.includes("king unprotected") ? [key] : [];
    }));
  }, [interactionPiece, playableMoveKeys, reachableMoves, state]);

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

  useEffect(() => {
    if (!energyView) setEnergySelection(null);
  }, [energyView]);

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
    if (energyView) {
      setEnergySelection(position);
      return;
    }
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
      {energyView && (
        <div className="energy-toolbar" aria-label="CMYK energy channels">
          <strong>CMYK ENERGY</strong>
          <div className="energy-channel-controls">
            {ENERGY_CHANNELS.map(({ pieceType, letter, channel }) => (
              <button
                type="button"
                key={pieceType}
                className={`${channel} ${energyChannels[pieceType] ? "active" : ""}`}
                title={`${pieceType} ${channel} channel`}
                aria-label={`${pieceType} energy channel`}
                aria-pressed={energyChannels[pieceType]}
                onClick={() => onToggleEnergyChannel(pieceType)}
              >
                {letter}
              </button>
            ))}
          </div>
        </div>
      )}
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
              const energy = energyGrid[y][x];
              const energySummary = energyView
                ? ` CMYK energy: ${ENERGY_CHANNELS.map(({ pieceType, letter }) => `${letter} ${Math.round(energy.ratios[pieceType] * 100)} percent`).join(", ")}.`
                : "";
              return (
                <Square
                  key={`${x}-${y}`}
                  position={position}
                  territory={projectFieldValue(displayField[y][x])}
                  fieldValue={displayField[y][x]}
                  piece={piece}
                  legal={!energyView && legalMoves.some((move) => samePosition(move, position))}
                  risky={!energyView && riskyMoveKeys.has(`${x}:${y}`)}
                  kingBlocked={!energyView && kingBlockedMoveKeys.has(`${x}:${y}`)}
                  selected={Boolean(piece && interactionPiece && piece.id === interactionPiece.id)}
                  dragging={piece?.id === draggingPieceId}
                  dragPreview={Boolean(draggingPieceId && dragPreview && samePosition(dragPreview, position))}
                  influenceTerritory={!energyView && influenceGrid ? projectFieldValue(influenceValue) : null}
                  influenceOpacity={!energyView && maximumInfluence > 0 && influence > 0
                    ? 0.45 + (influence / maximumInfluence) * 0.55
                    : 0}
                  highContrast={highContrast}
                  typeSums={!energyView && showTypeSums ? {
                    pawn: displayTypeFields.pawn[y][x],
                    rook: displayTypeFields.rook[y][x],
                    spy: displayTypeFields.spy[y][x],
                    king: displayTypeFields.king[y][x],
                  } : null}
                  lossPop={lossPopKeys.has(`${x}:${y}`)}
                  energyColor={energyView ? energy.color : undefined}
                  energySummary={energySummary}
                  energySelected={Boolean(energyView && energySelection && samePosition(energySelection, position))}
                  onClick={() => handleSquare(position)}
                />
              );
            }),
          )}
        </div>
        <div className="ranks right">{Array.from({ length: BOARD_SIZE }, (_, i) => <span key={i}>{BOARD_SIZE - i}</span>)}</div>
      </div>
      <div className="files bottom">{Array.from({ length: BOARD_SIZE }, (_, x) => <span key={x}>{x + 1}</span>)}</div>
      {energyView && energySelection && selectedEnergy && (
        <div className="energy-readout" aria-live="polite">
          <strong>SQUARE {energySelection.x + 1},{BOARD_SIZE - energySelection.y}</strong>
          <span>Intensity {Math.round(selectedEnergy.intensity * 100)}%</span>
          <div>
            {ENERGY_CHANNELS.map(({ pieceType, letter, channel }) => (
              <span className={!energyChannels[pieceType] ? "disabled" : ""} key={pieceType}>
                <i className={channel}>{letter}</i>
                <b>{Math.round(selectedEnergy.ratios[pieceType] * 100)}%</b>
                <small>{selectedEnergy.raw[pieceType] >= 0 ? "+" : ""}{selectedEnergy.raw[pieceType].toFixed(2)}</small>
              </span>
            ))}
          </div>
        </div>
      )}
      {!energyView && selectedPiece?.unstable && (
        <div className="piece-alert-hint">
          <div role="status" aria-live="polite">
            <strong>{selectedPiece.type === "king" ? "UNPROTECTED KING" : "UNSTABLE PIECE"}</strong>
            <p>
              {selectedPiece.type === "king"
                ? "Try alternate component tuning to create a legal escape, then move any piece that leaves the king on friendly or neutral territory."
                : "Move this piece or tune the field until its square is friendly or neutral. Otherwise it disappears when the turn ends."}
            </p>
          </div>
          {selectedPiece.type === "king" && selectedPiece.owner === state.currentPlayer && (
            <div className="check-actions">
              <button type="button" className="hint-button" disabled={hintSearching} onClick={onHint}>
                <Lightbulb size={15} aria-hidden="true" />
                {hintSearching ? "Searching..." : "Hint"}
              </button>
              <button type="button" className="resign-button" onClick={onResign}>
                <Flag size={15} aria-hidden="true" />
                Resign
              </button>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
