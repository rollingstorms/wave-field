import type { Piece as PieceModel, Position, Territory } from "../game/types";
import { Piece } from "./Piece";

interface SquareProps {
  position: Position;
  territory: Territory;
  fieldValue: number;
  piece?: PieceModel;
  legal: boolean;
  selected: boolean;
  highContrast: boolean;
  onClick: () => void;
}

export function Square({ position, territory, fieldValue, piece, legal, selected, highContrast, onClick }: SquareProps) {
  const marker = territory === "red" ? "+" : territory === "blue" ? "-" : "0";
  return (
    <button
      className={`square ${territory} ${legal ? "legal" : ""} ${selected ? "selected-square" : ""}`}
      onClick={onClick}
      aria-label={`Square ${position.x + 1},${position.y + 1}. ${territory} territory. Field ${fieldValue.toFixed(3)}`}
    >
      {highContrast && <span className="territory-marker">{marker}</span>}
      {legal && <span className="legal-dot" />}
      {piece && <Piece piece={piece} selected={selected} />}
      {piece?.unstable && <span className="unstable" aria-label="unstable">!</span>}
    </button>
  );
}
