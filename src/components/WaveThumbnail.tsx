import { memo } from "react";
import { BOARD_CENTER } from "../game/constants";
import { contributionGrid } from "../field/evaluateField";
import type { GameState, Piece, PieceType, Player } from "../game/types";
import { ContinuousPattern } from "./ContinuousPattern";

interface WaveThumbnailProps {
  state: GameState;
  player: Player;
  pieceType: PieceType;
}

function WaveThumbnailView({ state, player, pieceType }: WaveThumbnailProps) {
  const previewPiece: Piece = {
    id: `${player}-${pieceType}-preview`,
    owner: player,
    type: pieceType,
    position: { x: BOARD_CENTER, y: BOARD_CENTER },
    unstable: false,
  };
  const values = contributionGrid(previewPiece, state);
  return (
    <div className="wave-thumbnail" aria-label={`${player} ${pieceType} ${state.variant === "sparse" ? "support" : "wave"} preview`}>
      {state.variant === "sparse" ? (
        <div className="sparse-thumbnail-grid" aria-hidden="true">
          {values.flat().map((value, index) => (
            <span key={index} className={`sparse-thumbnail-cell ${Math.abs(value) >= 2 ? "strong" : Math.abs(value) >= 1 ? "weak" : "empty"} ${player}`} />
          ))}
        </div>
      ) : <ContinuousPattern values={values} samplesPerSquare={4} />}
      <span className="wave-thumbnail-origin" aria-hidden="true" />
    </div>
  );
}

export const WaveThumbnail = memo(WaveThumbnailView, (previous, next) => (
  previous.player === next.player
  && previous.pieceType === next.pieceType
  && previous.state.components[previous.player][previous.pieceType] === next.state.components[next.player][next.pieceType]
  && previous.state.definitions[previous.pieceType] === next.state.definitions[next.pieceType]
  && previous.state.waveScales[previous.pieceType] === next.state.waveScales[next.pieceType]
  && previous.state.homeEnergy[previous.pieceType] === next.state.homeEnergy[next.pieceType]
  && previous.state.ampSquares === next.state.ampSquares
  && previous.state.variant === next.state.variant
));
