import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Flag, Search } from "lucide-react";
import type {
  CSSProperties,
  MouseEvent as ReactMouseEvent,
  PointerEvent as ReactPointerEvent,
  TouchEvent as ReactTouchEvent,
} from "react";
import { BOARD_SIZE } from "../game/constants";
import { createCmykEnergyGrid, ENERGY_CHANNELS } from "../field/cmykEnergy";
import type { EnergyChannelState } from "../field/cmykEnergy";
import { continuousFieldColor } from "../field/continuousColor";
import { contributionGrid, evaluateContinuousField, evaluateField, evaluateTypeFields } from "../field/evaluateField";
import type { TypeFields } from "../field/evaluateField";
import { projectFieldValue } from "../field/projection";
import { getContinuousLegalMoves, getLegalMoves, getPieceAt, getPieceAtPrecise, samePosition, samePrecisePosition } from "../game/movement";
import { applyContinuousMove, applyMove, getContinuousPlayableMoves, getPlayableMoves } from "../game/rules";
import { PIECE_DISPLAY_NAMES, PIECE_INITIALS } from "../game/pieceLabels";
import type { GameState, Position, PrecisePosition } from "../game/types";
import { isAmpSquare } from "../game/variants";
import { markInstability } from "../game/victory";
import { Piece, PieceShape } from "./Piece";
import { Square } from "./Square";

interface BoardProps {
  state: GameState;
  field: number[][];
  typeFields: TypeFields;
  continuousField: boolean;
  showTypeSums: boolean;
  energyView: boolean;
  energyChannels: EnergyChannelState;
  locked?: boolean;
  onSelect: (pieceId: string | null) => void;
  onMove: (pieceId: string, destination: PrecisePosition) => void;
  onResign: () => void;
  onHint: (focusedPieceId?: string | null) => void;
  hintSearching?: boolean;
  onToggleEnergyChannel: (pieceType: keyof EnergyChannelState) => void;
}

const FILE_LABELS = Array.from({ length: BOARD_SIZE }, (_, index) => String.fromCharCode(65 + index));
const RANK_LABELS = Array.from({ length: BOARD_SIZE }, (_, index) => index + 1);
const CONTINUOUS_SAMPLES_PER_SQUARE = 9;

interface ActiveDrag {
  pieceId: string;
  contactId: number;
  input: "pointer" | "touch";
  start: PrecisePosition;
  legalMoves: PrecisePosition[];
}

interface LossPop {
  id: string;
  position: PrecisePosition;
}

interface MovementAnimation {
  id: string;
  piece: NonNullable<GameState["pieces"][number]>;
  from: PrecisePosition;
  to: PrecisePosition;
}

function visualY(y: number) {
  return BOARD_SIZE - 1 - y;
}

function playerLabel(player: GameState["currentPlayer"]) {
  return player === "blue" ? "Blue" : "Red";
}

function preciseKey(position: PrecisePosition) {
  return `${position.x.toFixed(6)}:${position.y.toFixed(6)}`;
}

export function Board({ state, field, typeFields, continuousField, showTypeSums, energyView, energyChannels, locked = false, onSelect, onMove, onResign, onHint, hintSearching = false, onToggleEnergyChannel }: BoardProps) {
  const boardRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<ActiveDrag | null>(null);
  const previousPiecesRef = useRef(state.pieces);
  const lossPopTimersRef = useRef<Array<ReturnType<typeof globalThis.setTimeout>>>([]);
  const suppressClickRef = useRef(false);
  const [draggingPieceId, setDraggingPieceId] = useState<string | null>(null);
  const [dragPreview, setDragPreview] = useState<PrecisePosition | null>(null);
  const [lossPops, setLossPops] = useState<LossPop[]>([]);
  const [movementAnimations, setMovementAnimations] = useState<MovementAnimation[]>([]);
  const [movingPieceIds, setMovingPieceIds] = useState<Set<string>>(() => new Set());
  const [energySelection, setEnergySelection] = useState<Position | null>(null);
  const selectedPiece = state.pieces.find((piece) => piece.id === state.selectedPieceId);
  const continuousInteraction = state.variant === "continuous" && !energyView;
  const interactionPiece = energyView ? undefined : state.pieces.find((piece) => piece.id === draggingPieceId) ?? selectedPiece;
  const reachableMoves = useMemo(
    () => !locked && interactionPiece
      ? continuousInteraction ? getContinuousLegalMoves(interactionPiece.id, state) : getLegalMoves(interactionPiece.id, state, field)
      : [],
    [continuousInteraction, field, interactionPiece, locked, state],
  );
  const playableMoves = useMemo(
    () => !locked && interactionPiece
      ? continuousInteraction ? getContinuousPlayableMoves(interactionPiece.id, state) : getPlayableMoves(interactionPiece.id, state, field)
      : [],
    [continuousInteraction, field, interactionPiece, locked, state],
  );
  const previewState = useMemo<GameState>(() => {
    if (!draggingPieceId || !dragPreview) return state;
    const piece = state.pieces.find((candidate) => candidate.id === draggingPieceId);
    if (piece && !samePrecisePosition(piece.position, dragPreview)) {
      const result = continuousInteraction
        ? applyContinuousMove(draggingPieceId, dragPreview, state)
        : applyMove(draggingPieceId, dragPreview, state);
      if (result.ok) return result.state;
    }
    const moved = {
      ...state,
      pieces: state.pieces.map((piece) =>
        piece.id === draggingPieceId ? { ...piece, position: dragPreview } : piece,
      ),
    };
    return markInstability(moved, evaluateField(moved));
  }, [continuousInteraction, dragPreview, draggingPieceId, state]);
  const previewing = Boolean(draggingPieceId && dragPreview);
  const displayField = useMemo(
    () => previewing ? evaluateField(previewState) : field,
    [field, previewState, previewing],
  );
  const displayTypeFields = useMemo(
    () => previewing ? evaluateTypeFields(previewState) : typeFields,
    [previewState, previewing, typeFields],
  );
  const maximumFieldMagnitude = Math.max(
    ...displayField.flat().map((value) => Math.abs(value)),
    0,
  );
  const continuousSamples = useMemo(
    () => continuousField && !energyView ? evaluateContinuousField(previewState, CONTINUOUS_SAMPLES_PER_SQUARE) : null,
    [continuousField, energyView, previewState],
  );
  const maximumContinuousMagnitude = continuousSamples
    ? Math.max(...continuousSamples.flat().map((sample) => Math.abs(sample.value)), 0)
    : 0;
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
    () => new Set(lossPops.map((pop) => preciseKey(pop.position))),
    [lossPops],
  );
  const riskyMoveLossCounts = useMemo(() => {
    if (!interactionPiece) return new Map<string, number>();
    const ownPieceIds = new Set(state.pieces.filter((piece) => piece.owner === interactionPiece.owner).map((piece) => piece.id));
    return new Map(playableMoves.flatMap((move) => {
      const result = continuousInteraction
        ? applyContinuousMove(interactionPiece.id, move, state, { analyzeCheckmate: false })
        : applyMove(interactionPiece.id, move, state, { analyzeCheckmate: false });
      if (!result.ok) return [];
      const survivingOwnIds = new Set(result.state.pieces.filter((piece) => piece.owner === interactionPiece.owner).map((piece) => piece.id));
      const lossCount = [...ownPieceIds].filter((id) => !survivingOwnIds.has(id)).length;
      return lossCount > 0 ? [[preciseKey(move), lossCount] as const] : [];
    }));
  }, [continuousInteraction, interactionPiece, playableMoves, state]);
  const legalMoves = useMemo(() => {
    const safeMoves = playableMoves.filter((move) => !riskyMoveLossCounts.has(preciseKey(move)));
    if (safeMoves.length > 0 || playableMoves.length === 0) return playableMoves;
    const minimumLoss = Math.min(...playableMoves.map((move) => riskyMoveLossCounts.get(preciseKey(move)) ?? 0));
    return playableMoves.filter((move) => (riskyMoveLossCounts.get(preciseKey(move)) ?? 0) === minimumLoss);
  }, [playableMoves, riskyMoveLossCounts]);
  const riskyMoveKeys = useMemo(
    () => new Set(legalMoves.flatMap((move) => riskyMoveLossCounts.has(preciseKey(move)) ? [preciseKey(move)] : [])),
    [legalMoves, riskyMoveLossCounts],
  );
  const playableMoveKeys = useMemo(
    () => new Set(playableMoves.map(preciseKey)),
    [playableMoves],
  );
  const kingBlockedMoveKeys = useMemo(() => {
    if (continuousInteraction) return new Set<string>();
    if (!interactionPiece) return new Set<string>();
    return new Set(reachableMoves.flatMap((move) => {
      const key = preciseKey(move);
      if (playableMoveKeys.has(key)) return [];
      const result = applyMove(interactionPiece.id, move, state, { analyzeCheckmate: false });
      return result.reason?.toLowerCase().includes("big hat unprotected") || result.reason?.toLowerCase().includes("king unprotected") ? [key] : [];
    }));
  }, [continuousInteraction, interactionPiece, playableMoveKeys, reachableMoves, state]);
  const safeMoves = useMemo(
    () => playableMoves.filter((move) => !riskyMoveLossCounts.has(preciseKey(move))),
    [playableMoves, riskyMoveLossCounts],
  );
  const noSafePlayableMoves = playableMoves.length > 0 && safeMoves.length === 0;
  const bigHatNeedsHint = selectedPiece?.type === "king"
    && selectedPiece.owner === state.currentPlayer
    && selectedPiece.unstable;
  const onlyBigHatBlockedMoves = reachableMoves.length > 0
    && playableMoves.length === 0
    && kingBlockedMoveKeys.size === reachableMoves.length;
  const showStuckHint = Boolean(
    !locked
      && !energyView
      && !draggingPieceId
      && selectedPiece
      && selectedPiece.owner === state.currentPlayer
      && (noSafePlayableMoves || onlyBigHatBlockedMoves || bigHatNeedsHint),
  );

  useLayoutEffect(() => {
    const previousPieces = previousPiecesRef.current;
    const previousById = new Map(previousPieces.map((piece) => [piece.id, piece]));
    const currentIds = new Set(state.pieces.map((piece) => piece.id));
    const lost = previousPieces.filter((piece) => !currentIds.has(piece.id));
    const moved = state.pieces.flatMap((piece) => {
      const previous = previousById.get(piece.id);
      return previous && !samePrecisePosition(previous.position, piece.position)
        ? [{ id: `${piece.id}:${globalThis.performance.now()}`, piece, from: previous.position, to: piece.position }]
        : [];
    });
    previousPiecesRef.current = state.pieces;

    if (moved.length > 0) {
      const movedIds = new Set(moved.map((animation) => animation.piece.id));
      setMovementAnimations((animations) => [...animations, ...moved]);
      setMovingPieceIds((ids) => new Set([...ids, ...movedIds]));
      const timer = globalThis.setTimeout(() => {
        setMovementAnimations((animations) => animations.filter((animation) => !moved.some((candidate) => candidate.id === animation.id)));
        setMovingPieceIds((ids) => {
          const next = new Set(ids);
          for (const id of movedIds) next.delete(id);
          return next;
        });
        lossPopTimersRef.current = lossPopTimersRef.current.filter((candidate) => candidate !== timer);
      }, 260);
      lossPopTimersRef.current.push(timer);
    }

    if (lost.length > 0) {
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
    }
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
    const visualRow = Math.floor(((clientY - bounds.top) / bounds.height) * BOARD_SIZE);
    const y = visualY(visualRow);
    const position = { x, y };
    return x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE ? position : null;
  }

  function precisePositionFromPointer(clientX: number, clientY: number): PrecisePosition | null {
    const board = boardRef.current;
    if (!board) return null;
    const bounds = board.getBoundingClientRect();
    const rawX = ((clientX - bounds.left) / bounds.width) * BOARD_SIZE - 0.5;
    const rawVisualY = ((clientY - bounds.top) / bounds.height) * BOARD_SIZE - 0.5;
    const rawY = visualY(rawVisualY);
    const step = 1 / CONTINUOUS_SAMPLES_PER_SQUARE;
    const x = Math.min(BOARD_SIZE - 1, Math.max(0, Math.round(rawX / step) * step));
    const y = Math.min(BOARD_SIZE - 1, Math.max(0, Math.round(rawY / step) * step));
    return Number.isFinite(x) && Number.isFinite(y) ? { x, y } : null;
  }

  function getContinuousPieceNear(position: PrecisePosition): GameState["pieces"][number] | undefined {
    const exact = getPieceAtPrecise(state, position);
    if (exact) return exact;
    const pickRadius = 0.5;
    return state.pieces.find((piece) =>
      Math.abs(piece.position.x - position.x) <= pickRadius
      && Math.abs(piece.position.y - position.y) <= pickRadius);
  }

  function displayedPlayableMovesFor(pieceId: string): PrecisePosition[] {
    if (continuousInteraction) return getContinuousPlayableMoves(pieceId, state);
    const piece = state.pieces.find((candidate) => candidate.id === pieceId);
    if (!piece) return [];
    const moves = getPlayableMoves(pieceId, state, field);
    const ownPieceIds = new Set(state.pieces.filter((candidate) => candidate.owner === piece.owner).map((candidate) => candidate.id));
    const lossCounts = new Map(moves.flatMap((move) => {
      const result = applyMove(pieceId, move, state, { analyzeCheckmate: false });
      if (!result.ok) return [];
      const survivingOwnIds = new Set(result.state.pieces.filter((candidate) => candidate.owner === piece.owner).map((candidate) => candidate.id));
      const lossCount = [...ownPieceIds].filter((id) => !survivingOwnIds.has(id)).length;
      return lossCount > 0 ? [[`${move.x}:${move.y}`, lossCount] as const] : [];
    }));
    const safe = moves.filter((move) => !lossCounts.has(`${move.x}:${move.y}`));
    if (safe.length > 0 || moves.length === 0) return moves;
    const minimumLoss = Math.min(...moves.map((move) => lossCounts.get(`${move.x}:${move.y}`) ?? 0));
    return moves.filter((move) => (lossCounts.get(`${move.x}:${move.y}`) ?? 0) === minimumLoss);
  }

  function startDrag(contactId: number, input: ActiveDrag["input"], clientX: number, clientY: number) {
    if (locked || state.status !== "playing" || dragRef.current) return false;
    const precisePosition = continuousInteraction ? precisePositionFromPointer(clientX, clientY) : null;
    if (
      continuousInteraction
      && selectedPiece
      && precisePosition
      && !samePrecisePosition(selectedPiece.position, precisePosition)
    ) {
      return false;
    }
    const squarePosition = precisePosition ? null : positionFromPointer(clientX, clientY);
    const piece = precisePosition ? getContinuousPieceNear(precisePosition) : squarePosition ? getPieceAt(state, squarePosition) : undefined;
    if (!piece || piece.owner !== state.currentPlayer) return false;

    const moves = displayedPlayableMovesFor(piece.id);
    dragRef.current = { pieceId: piece.id, contactId, input, start: piece.position, legalMoves: moves };
    setDraggingPieceId(piece.id);
    setDragPreview(piece.position);
    onSelect(piece.id);
    return true;
  }

  function updateDrag(contactId: number, input: ActiveDrag["input"], clientX: number, clientY: number) {
    const drag = dragRef.current;
    if (!drag || drag.contactId !== contactId || drag.input !== input) return false;
    const position = continuousInteraction ? precisePositionFromPointer(clientX, clientY) : positionFromPointer(clientX, clientY);
    const legal = position && drag.legalMoves.some((move) => samePrecisePosition(move, position));
    const nextPreview = legal ? position : drag.start;
    setDragPreview((current) => current && samePrecisePosition(current, nextPreview) ? current : nextPreview);
    return true;
  }

  function completeDrag(contactId: number, input: ActiveDrag["input"], clientX: number, clientY: number, commit: boolean) {
    const drag = dragRef.current;
    if (!drag || drag.contactId !== contactId || drag.input !== input) return false;
    const position = continuousInteraction ? precisePositionFromPointer(clientX, clientY) : positionFromPointer(clientX, clientY);
    const destination = commit && position && drag.legalMoves.some((move) => samePrecisePosition(move, position))
      ? position
      : null;

    if (destination && !samePrecisePosition(destination, drag.start)) {
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

  function handleSquare(position: Position, event: ReactMouseEvent<HTMLButtonElement>) {
    if (energyView) {
      setEnergySelection(position);
      return;
    }
    if (locked) return;
    if (continuousInteraction) {
      const precisePosition = precisePositionFromPointer(event.clientX, event.clientY);
      if (!precisePosition) return;
      const clickedPiece = getContinuousPieceNear(precisePosition);
      const isLegalPoint = selectedPiece && legalMoves.some((move) => samePrecisePosition(move, precisePosition));
      if (selectedPiece && isLegalPoint) {
        onMove(selectedPiece.id, precisePosition);
        return;
      }
      if (clickedPiece?.owner === state.currentPlayer && state.status === "playing") {
        onSelect(clickedPiece.id);
        return;
      }
      onSelect(null);
      return;
    }
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

  function handleContinuousBoardClick(event: ReactMouseEvent<HTMLButtonElement>) {
    if (locked || energyView || !continuousInteraction) return;
    const precisePosition = precisePositionFromPointer(event.clientX, event.clientY);
    if (!precisePosition) return;
    const clickedPiece = getContinuousPieceNear(precisePosition);
    const isLegalPoint = selectedPiece && legalMoves.some((move) => samePrecisePosition(move, precisePosition));
    if (selectedPiece && isLegalPoint) {
      onMove(selectedPiece.id, precisePosition);
      return;
    }
    if (clickedPiece?.owner === state.currentPlayer && state.status === "playing") {
      onSelect(clickedPiece.id);
      return;
    }
    onSelect(null);
  }

  const continuousPieces = continuousInteraction ? previewState.pieces : [];

  return (
    <section className="board-wrap" aria-label="Wave Field board" style={{ "--board-size": BOARD_SIZE } as CSSProperties}>
      {energyView && (
        <div className="energy-toolbar" aria-label="CMYK energy channels">
          <strong>CMYK ENERGY</strong>
          <div className="energy-channel-controls">
            {ENERGY_CHANNELS.map(({ pieceType, channel }) => (
              <button
                type="button"
                key={pieceType}
                className={`${channel} ${energyChannels[pieceType] ? "active" : ""}`}
                title={`${PIECE_DISPLAY_NAMES[pieceType]} ${channel} channel`}
                aria-label={`${PIECE_DISPLAY_NAMES[pieceType]} energy channel`}
                aria-pressed={energyChannels[pieceType]}
                onClick={() => onToggleEnergyChannel(pieceType)}
              >
                <svg className="energy-piece-shape" viewBox="0 0 64 64" aria-hidden="true">
                  <PieceShape type={pieceType} />
                </svg>
              </button>
            ))}
          </div>
        </div>
      )}
      {showTypeSums && (
        <div className="type-sum-key" aria-label="Type sum corner key">
          <strong>TYPE SUMS</strong>
          <span>{PIECE_INITIALS.pawn} ↖</span>
          <span>{PIECE_INITIALS.rook} ↗</span>
          <span>{PIECE_INITIALS.spy} ↙</span>
          <span>{PIECE_INITIALS.king} ↘</span>
        </div>
      )}
      <div className="files top">{FILE_LABELS.map((file) => <span key={file}>{file}</span>)}</div>
      <div className="board-row-wrap">
        <div className="ranks left">{RANK_LABELS.map((rank) => <span key={rank}>{rank}</span>)}</div>
        <div
          className={`board ${draggingPieceId ? "dragging" : ""} ${continuousInteraction || continuousSamples ? "continuous-render" : ""}`}
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
          {continuousSamples && (
            <div
              className="continuous-field-layer"
              style={{
                "--continuous-resolution": BOARD_SIZE * CONTINUOUS_SAMPLES_PER_SQUARE,
              } as CSSProperties}
              aria-hidden="true"
            >
              {continuousSamples.flatMap((row, rowIndex) =>
                row.map((sample, x) => (
                  <span
                    key={`${x}-${rowIndex}`}
                    style={{ backgroundColor: continuousFieldColor(sample.value, maximumContinuousMagnitude) }}
                  />
                )),
              )}
            </div>
          )}
          {continuousInteraction && (
            <button
              type="button"
              className="continuous-hit-layer"
              aria-label="Continuous movement field"
              onClick={handleContinuousBoardClick}
            />
          )}
          {movementAnimations.map((animation) => (
            <span
              className="piece-move-ghost"
              key={animation.id}
              style={{
                "--from-x": animation.from.x,
                "--from-y": visualY(animation.from.y),
                "--to-x": animation.to.x,
                "--to-y": visualY(animation.to.y),
                transform: `translate(${animation.from.x * 100}%, ${visualY(animation.from.y) * 100}%)`,
              } as CSSProperties}
              aria-hidden="true"
            >
              <Piece piece={animation.piece} selected={false} dragging={false} />
            </span>
          ))}
          {continuousInteraction && interactionPiece && legalMoves.map((move) => (
            <span
              className={`continuous-legal-point ${riskyMoveKeys.has(preciseKey(move)) ? "risky-point" : ""}`}
              key={preciseKey(move)}
              style={{
                "--move-x": move.x,
                "--move-y": visualY(move.y),
              } as CSSProperties}
              aria-hidden="true"
            />
          ))}
          {continuousPieces.map((piece) => (
            <span
              className="continuous-piece-anchor"
              key={piece.id}
              data-piece-id={piece.id}
              role="img"
              aria-label={`${playerLabel(piece.owner)} ${PIECE_DISPLAY_NAMES[piece.type]}${interactionPiece?.id === piece.id ? ", selected" : ""}${piece.unstable ? ", unstable" : ""}`}
              style={{
                "--piece-x": piece.position.x,
                "--piece-y": visualY(piece.position.y),
              } as CSSProperties}
            >
              <Piece
                piece={piece}
                selected={Boolean(interactionPiece && piece.id === interactionPiece.id)}
                dragging={piece.id === draggingPieceId}
                hidden={Boolean(movingPieceIds.has(piece.id))}
              />
              {piece.unstable && <span className="unstable continuous-unstable" aria-label="unstable">!</span>}
            </span>
          ))}
          {Array.from({ length: BOARD_SIZE }, (_, row) =>
            Array.from({ length: BOARD_SIZE }, (_, x) => {
              const y = visualY(row);
              const position = { x, y };
              const piece = continuousInteraction ? undefined : getPieceAt(previewState, position);
              const influenceValue = influenceGrid?.[y][x] ?? 0;
              const influence = Math.abs(influenceValue);
              const energy = energyGrid[y][x];
              const energySummary = energyView
                ? ` CMYK energy: ${ENERGY_CHANNELS.map(({ pieceType }) => `${PIECE_DISPLAY_NAMES[pieceType]} ${Math.round(energy.ratios[pieceType] * 100)} percent`).join(", ")}.`
                : "";
              const fieldMagnitudePercent = maximumFieldMagnitude > 0
                ? Math.round((Math.abs(displayField[y][x]) / maximumFieldMagnitude) * 100)
                : 0;
              return (
                <Square
                  key={`${x}-${y}`}
                  position={position}
                  territory={projectFieldValue(displayField[y][x])}
                  fieldValue={displayField[y][x]}
                  piece={piece}
                  legal={!energyView && !continuousInteraction && legalMoves.some((move) => samePosition(move, position))}
                  risky={!energyView && riskyMoveKeys.has(preciseKey(position))}
                  kingBlocked={!energyView && kingBlockedMoveKeys.has(preciseKey(position))}
                  selected={Boolean(piece && interactionPiece && piece.id === interactionPiece.id)}
                  dragging={piece?.id === draggingPieceId}
                  dragPreview={Boolean(!continuousInteraction && draggingPieceId && dragPreview && samePosition(dragPreview, position))}
                  influenceTerritory={!energyView && influenceGrid ? projectFieldValue(influenceValue) : null}
                  influenceOpacity={!energyView && maximumInfluence > 0 && influence > 0
                    ? 0.45 + (influence / maximumInfluence) * 0.55
                    : 0}
                  typeSums={!energyView && showTypeSums ? {
                    pawn: displayTypeFields.pawn[y][x],
                    rook: displayTypeFields.rook[y][x],
                    spy: displayTypeFields.spy[y][x],
                    king: displayTypeFields.king[y][x],
                  } : null}
                  amp={isAmpSquare(position, state.ampSquares)}
                  lossPop={lossPopKeys.has(preciseKey(position))}
                  energyColor={energyView ? energy.color : undefined}
                  energySummary={energySummary}
                  energySelected={Boolean(energyView && energySelection && samePosition(energySelection, position))}
                  continuousColor={continuousField && !energyView && !continuousSamples
                    ? continuousFieldColor(displayField[y][x], maximumFieldMagnitude)
                    : undefined}
                  continuousSummary={continuousField && !energyView
                    ? ` Relative field magnitude ${fieldMagnitudePercent} percent.`
                    : ""}
                  hidePiece={Boolean(piece && movingPieceIds.has(piece.id))}
                  passive={continuousInteraction}
                  onClick={(event) => handleSquare(position, event)}
                />
              );
            }),
          )}
          {showStuckHint && selectedPiece && (
            <button
              type="button"
              className="stuck-hint-button"
              style={{
                "--hint-x": selectedPiece.position.x,
                "--hint-y": visualY(selectedPiece.position.y),
              } as CSSProperties}
              disabled={hintSearching}
              title="Find hint"
              aria-label="Find hint"
              onPointerDown={(event) => event.stopPropagation()}
              onTouchStart={(event) => event.stopPropagation()}
              onClick={(event) => {
                event.stopPropagation();
                onHint(selectedPiece.id);
              }}
            >
              <Search size={20} aria-hidden="true" />
            </button>
          )}
        </div>
        <div className="ranks right">{RANK_LABELS.map((rank) => <span key={rank}>{rank}</span>)}</div>
      </div>
      <div className="files bottom">{FILE_LABELS.map((file) => <span key={file}>{file}</span>)}</div>
      {energyView && energySelection && selectedEnergy && (
        <div className="energy-readout" aria-live="polite">
          <strong>SQUARE {FILE_LABELS[energySelection.x]}{BOARD_SIZE - energySelection.y}</strong>
          <span>Intensity {Math.round(selectedEnergy.intensity * 100)}%</span>
          <div>
            {ENERGY_CHANNELS.map(({ pieceType, channel }) => (
              <span className={!energyChannels[pieceType] ? "disabled" : ""} key={pieceType}>
                <i className={channel}>
                  <svg className="energy-piece-shape" viewBox="0 0 64 64" aria-hidden="true">
                    <PieceShape type={pieceType} />
                  </svg>
                </i>
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
            <strong>{selectedPiece.type === "king" ? "UNPROTECTED BIG HAT" : "UNSTABLE PIECE"}</strong>
            <p>
              {selectedPiece.type === "king"
                ? "Try alternate component tuning to create a safe escape, then move any piece that leaves the Big Hat on friendly or neutral territory."
                : "Move this piece or tune the field until its square is friendly or neutral. Otherwise it disappears when the turn ends."}
            </p>
          </div>
          {selectedPiece.type === "king" && selectedPiece.owner === state.currentPlayer && (
            <div className="check-actions">
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
