import { COMPONENT_COUNTS, PIECE_STRENGTH } from "../game/constants";
import { canSetComponentValue, getTuningLoad } from "../game/tuning";
import type { Coefficient, GameState, PieceType } from "../game/types";
import { WaveThumbnail } from "./WaveThumbnail";

const pieceTypes: PieceType[] = ["pawn", "rook", "spy", "king"];
const values: Coefficient[] = [1, 0, -1];
const label = (value: Coefficient) => value === 1 ? "+" : value === -1 ? "-" : "0";

interface ComponentControlsProps {
  state: GameState;
  locked?: boolean;
  onTune: (pieceType: PieceType, componentIndex: number, value: Coefficient) => void;
}

export function ComponentControls({ state, locked = false, onTune }: ComponentControlsProps) {
  const player = state.currentPlayer;
  return (
    <aside className="control-panel" aria-label={`${player} component controls`}>
      <div className="legend">
        <span><i className="swatch red" /> &gt; 0</span>
        <span><i className="swatch neutral" /> = 0</span>
        <span><i className="swatch blue" /> &lt; 0</span>
      </div>
      {pieceTypes.map((pieceType) => (
        <section className="component-group" key={pieceType}>
          <div className="piece-heading">
            <WaveThumbnail state={state} player={player} pieceType={pieceType} />
            <div className="piece-heading-copy">
              <strong>{pieceType.toUpperCase()}</strong>
              <span>({COMPONENT_COUNTS[pieceType]} components)</span>
              <span>Strength {getTuningLoad(state.components[player][pieceType])}/{PIECE_STRENGTH[pieceType]}</span>
            </div>
          </div>
          {state.components[player][pieceType].map((coefficient, index) => (
            <div className="coefficient-row" key={`${pieceType}-${index}`}>
              <span>C{index + 1}</span>
              {values.map((value) => (
                <button
                  key={value}
                  type="button"
                  className={coefficient === value ? `chosen ${player}` : ""}
                  disabled={
                    locked
                    || state.status !== "playing"
                    || coefficient === value
                    || !canSetComponentValue(state.components[player], pieceType, index, value)
                  }
                  onClick={() => onTune(pieceType, index, value)}
                  aria-pressed={coefficient === value}
                  aria-label={`${player} ${pieceType} component ${index + 1} set to ${label(value)}`}
                >
                  {label(value)}
                </button>
              ))}
            </div>
          ))}
        </section>
      ))}
    </aside>
  );
}
