import type { Piece as PieceModel } from "../game/types";

interface PieceProps {
  piece: PieceModel;
  selected: boolean;
}

export function Piece({ piece, selected }: PieceProps) {
  const className = `piece ${piece.owner} ${piece.type} ${selected ? "selected" : ""}`;
  return (
    <svg className={className} viewBox="0 0 64 64" aria-label={`${piece.owner} ${piece.type}`}>
      {piece.type === "pawn" && <path d="M20 52V25a12 12 0 0 1 24 0v27Z" />}
      {piece.type === "spy" && <path d="M20 52V29L32 10l12 19v23Z" />}
      {piece.type === "rook" && <path d="M16 52V16h8v8h8v-8h8v8h8v28Z" />}
      {piece.type === "king" && (
        <>
          <path d="M14 52a18 18 0 0 1 36 0Z" />
          <path className="accent" d="M32 11v17M24 19h16" />
        </>
      )}
    </svg>
  );
}
