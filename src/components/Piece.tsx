import type { Piece as PieceModel } from "../game/types";
import { pieceNameLower } from "../game/pieceLabels";

interface PieceProps {
  piece: PieceModel;
  selected: boolean;
  dragging: boolean;
  hidden?: boolean;
}

const paths = {
  pawn: "M18 48V31c0-7.7 6.3-14 14-14s14 6.3 14 14v17Z",
  rook: "M18 10h28v44H18Z",
  spy: "M32 17 46 48H18Z",
} as const;

const transforms = {
  pawn: "",
  rook: "",
  spy: "",
} as const;

export function Piece({ piece, selected, dragging, hidden = false }: PieceProps) {
  const className = `piece ${piece.owner} ${piece.type} ${selected ? "selected" : ""} ${dragging ? "dragging" : ""} ${hidden ? "moving-hidden" : ""}`;
  return (
    <svg className={className} viewBox="0 0 64 64" aria-label={`${piece.owner} ${pieceNameLower(piece.type)}`}>
      {piece.type === "king" ? (
        <>
          <path d="M20 47V20a12 12 0 0 1 24 0v27Z" />
          <path d="M8 47h48v7H8Z" />
        </>
      ) : (
        <path d={paths[piece.type]} transform={transforms[piece.type]} />
      )}
    </svg>
  );
}
