import type { Piece as PieceModel } from "../game/types";

interface PieceProps {
  piece: PieceModel;
  selected: boolean;
  dragging: boolean;
}

export function Piece({ piece, selected, dragging }: PieceProps) {
  const className = `piece ${piece.owner} ${piece.type} ${selected ? "selected" : ""} ${dragging ? "dragging" : ""}`;
  return (
    <svg className={className} viewBox="0 0 64 64" aria-label={`${piece.owner} ${piece.type}`}>
      {piece.type === "pawn" && <path d="M20 52V25a12 12 0 0 1 24 0v27Z" />}
      {piece.type === "spy" && <path d="M20 52V29L32 10l12 19v23Z" />}
      {piece.type === "rook" && <path d="M16 52V16h8v8h8v-8h8v8h8v28Z" />}
      {piece.type === "king" && (
        <>
          <path d="M18 46c0-6 2-17 4-24 2-6 6-11 12-13h12c6 4 10 11 10 19v18Z" />
          <path className="accent" d="M7 48h50" />
        </>
      )}
    </svg>
  );
}
