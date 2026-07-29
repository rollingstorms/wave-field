import { BOARD_SIZE, FIELD_EPSILON } from "../game/constants";
import { PIECE_DISPLAY_NAMES, PIECE_INITIALS } from "../game/pieceLabels";
import type { Piece as PieceModel, PieceType, Position, Territory } from "../game/types";
import { Piece } from "./Piece";

interface SquareProps {
  position: Position;
  territory: Territory;
  fieldValue: number;
  piece?: PieceModel;
  legal: boolean;
  risky: boolean;
  kingBlocked: boolean;
  selected: boolean;
  dragging: boolean;
  dragPreview: boolean;
  influenceTerritory: Territory | null;
  influenceOpacity: number;
  typeSums: Record<PieceType, number> | null;
  lossPop: boolean;
  energyColor?: string;
  energySummary?: string;
  energySelected?: boolean;
  continuousColor?: string;
  continuousSummary?: string;
  hidePiece?: boolean;
  onClick: () => void;
}

function formatSigned(value: number) {
  if (Math.abs(value) <= FIELD_EPSILON) return "0";
  const magnitude = Math.abs(value) >= 10 ? value.toFixed(0) : value.toFixed(1);
  return `${value > 0 ? "+" : ""}${magnitude.replace(".0", "")}`;
}

export function Square({ position, territory, fieldValue, piece, legal, risky, kingBlocked, selected, dragging, dragPreview, influenceTerritory, influenceOpacity, typeSums, lossPop, energyColor, energySummary = "", energySelected = false, continuousColor, continuousSummary = "", hidePiece = false, onClick }: SquareProps) {
  const renderedColor = energyColor ?? continuousColor;
  const coordinate = `${String.fromCharCode(65 + position.x)}${BOARD_SIZE - position.y}`;
  const influenceSummary = influenceTerritory
    ? ` Selected piece influence ${influenceTerritory}.`
    : "";
  const typeSummary = typeSums
    ? ` Total ${formatSigned(fieldValue)}. ${PIECE_DISPLAY_NAMES.pawn} ${formatSigned(typeSums.pawn)}, ${PIECE_DISPLAY_NAMES.rook} ${formatSigned(typeSums.rook)}, ${PIECE_DISPLAY_NAMES.spy} ${formatSigned(typeSums.spy)}, ${PIECE_DISPLAY_NAMES.king} ${formatSigned(typeSums.king)}.`
    : "";
  return (
    <button
      className={`square ${territory} ${energyColor ? "energy-square" : ""} ${continuousColor ? "continuous-square" : ""} ${energySelected ? "energy-selected" : ""} ${legal ? "legal" : ""} ${risky ? "risky-move" : ""} ${selected ? "selected-square" : ""} ${dragPreview ? "drag-preview-square" : ""} ${typeSums ? "type-sums-visible" : ""}`}
      style={renderedColor ? { backgroundColor: renderedColor } : undefined}
      onClick={onClick}
      data-board-x={position.x}
      data-board-y={position.y}
      aria-label={`Square ${coordinate}. ${territory} territory. Field ${fieldValue.toFixed(3)}.${energySummary}${continuousSummary}${risky ? " Moving here loses a piece." : ""}${kingBlocked ? " Reachable, but blocked because it would leave your Big Hat unprotected." : ""}${influenceSummary}${typeSummary}`}
    >
      {influenceOpacity > 0 && influenceTerritory !== "neutral" && (
        <span
          className={`influence-overlay influence-${influenceTerritory}`}
          style={{ opacity: influenceOpacity }}
          aria-hidden="true"
        />
      )}
      {(selected || dragPreview) && (
        <span className={`square-outline ${dragPreview ? "drag-outline" : "selection-outline"}`} aria-hidden="true" />
      )}
      {legal && <span className={risky ? "legal-dot risky-dot" : "legal-dot"} />}
      {risky && <span className="risk-marker" aria-hidden="true">!</span>}
      {kingBlocked && <span className="king-block-marker" aria-hidden="true">{PIECE_INITIALS.king}</span>}
      {piece && <Piece piece={piece} selected={selected} dragging={dragging} hidden={hidePiece} />}
      {piece?.unstable && <span className="unstable" aria-label="unstable">!</span>}
      {lossPop && (
        <span className="loss-pop" aria-hidden="true">
          {Array.from({ length: 8 }, (_, index) => <i key={index} />)}
        </span>
      )}
      {typeSums && (
        <span className="type-sums" aria-hidden="true">
          <i className="type-sum-value pawn">{formatSigned(typeSums.pawn)}</i>
          <i className="type-sum-value rook">{formatSigned(typeSums.rook)}</i>
          <i className="type-sum-value total">{formatSigned(fieldValue)}</i>
          <i className="type-sum-value spy">{formatSigned(typeSums.spy)}</i>
          <i className="type-sum-value king">{formatSigned(typeSums.king)}</i>
        </span>
      )}
    </button>
  );
}
