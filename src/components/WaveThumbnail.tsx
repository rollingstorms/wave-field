import { BOARD_CENTER } from "../game/constants";
import { contributionGrid } from "../field/evaluateField";
import type { GameState, Piece, PieceType, Player } from "../game/types";
import { ContinuousPattern } from "./ContinuousPattern";

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
  return (
    <div className="wave-thumbnail" aria-label={`${player} ${pieceType} wave preview`}>
      <ContinuousPattern values={values} samplesPerSquare={4} />
      <span className="wave-thumbnail-origin" aria-hidden="true" />
    </div>
  );
}
