import { BOARD_CENTER, FIELD_EPSILON } from "../game/constants";
import { contributionGrid } from "../field/evaluateField";
import type { GameState, Piece, PieceType, Player } from "../game/types";

interface WaveThumbnailProps {
  state: GameState;
  player: Player;
  pieceType: PieceType;
}

export function WaveThumbnail({ state, player, pieceType }: WaveThumbnailProps) {
  const previewPiece: Piece = {
    id: `${player}-${pieceType}-preview`,
    owner: player,
    type: pieceType,
    position: { x: BOARD_CENTER, y: BOARD_CENTER },
    unstable: false,
  };
  const values = contributionGrid(previewPiece, state);
  const maximum = Math.max(...values.flat().map(Math.abs), FIELD_EPSILON);

  return (
    <div className="wave-thumbnail" aria-label={`${player} ${pieceType} wave preview`}>
      {values.map((row, y) => row.map((value, x) => {
        const territory = value > FIELD_EPSILON ? "red" : value < -FIELD_EPSILON ? "blue" : "neutral";
        const intensity = territory === "neutral" ? 1 : 0.35 + (Math.abs(value) / maximum) * 0.65;
        return (
          <i
            className={`wave-thumbnail-cell ${territory} ${x === BOARD_CENTER && y === BOARD_CENTER ? "origin" : ""}`}
            key={`${x}-${y}`}
            style={{ opacity: intensity }}
            title={value.toFixed(3)}
            aria-hidden="true"
          />
        );
      }))}
    </div>
  );
}
