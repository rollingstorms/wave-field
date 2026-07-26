import { FIELD_EPSILON } from "../game/constants";
import type { Piece as PieceModel, PieceType, Position, Territory } from "../game/types";
import { Piece } from "./Piece";

interface SquareProps {
  position: Position;
  territory: Territory;
  fieldValue: number;
  piece?: PieceModel;
  legal: boolean;
  selected: boolean;
  dragging: boolean;
  dragPreview: boolean;
  influenceTerritory: Territory | null;
  influenceOpacity: number;
  highContrast: boolean;
  typeSums: Record<PieceType, number> | null;
  onClick: () => void;
}

function formatSigned(value: number) {
  if (Math.abs(value) <= FIELD_EPSILON) return "0";
  const magnitude = Math.abs(value) >= 10 ? value.toFixed(0) : value.toFixed(1);
  return `${value > 0 ? "+" : ""}${magnitude.replace(".0", "")}`;
}

export function Square({ position, territory, fieldValue, piece, legal, selected, dragging, dragPreview, influenceTerritory, influenceOpacity, highContrast, typeSums, onClick }: SquareProps) {
  const marker = territory === "red" ? "+" : territory === "blue" ? "-" : "0";
  const influenceSummary = influenceTerritory
    ? ` Selected piece influence ${influenceTerritory}.`
    : "";
  const typeSummary = typeSums
    ? ` Pawn ${formatSigned(typeSums.pawn)}, rook ${formatSigned(typeSums.rook)}, spy ${formatSigned(typeSums.spy)}, king ${formatSigned(typeSums.king)}.`
    : "";
  return (
    <button
      className={`square ${territory} ${legal ? "legal" : ""} ${selected ? "selected-square" : ""} ${dragPreview ? "drag-preview-square" : ""} ${typeSums ? "type-sums-visible" : ""}`}
      onClick={onClick}
      data-board-x={position.x}
      data-board-y={position.y}
      aria-label={`Square ${position.x + 1},${position.y + 1}. ${territory} territory. Field ${fieldValue.toFixed(3)}.${influenceSummary}${typeSummary}`}
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
      {highContrast && <span className="territory-marker">{marker}</span>}
      {legal && <span className="legal-dot" />}
      {piece && <Piece piece={piece} selected={selected} dragging={dragging} />}
      {piece?.unstable && <span className="unstable" aria-label="unstable">!</span>}
      {typeSums && (
        <span className="type-sums" aria-hidden="true">
          <i className="type-sum-value pawn">{formatSigned(typeSums.pawn)}</i>
          <i className="type-sum-value rook">{formatSigned(typeSums.rook)}</i>
          <i className="type-sum-value spy">{formatSigned(typeSums.spy)}</i>
          <i className="type-sum-value king">{formatSigned(typeSums.king)}</i>
        </span>
      )}
    </button>
  );
}
