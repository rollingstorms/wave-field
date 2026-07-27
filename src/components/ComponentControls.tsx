import { Dices, RotateCcw } from "lucide-react";
import { COMPONENT_COUNTS, TUNING_STRENGTH } from "../game/constants";
import { getTuningLoad } from "../game/tuning";
import type { Coefficient, GameState, Piece, PieceType, Player } from "../game/types";
import { Piece as PieceShape } from "./Piece";
import { WaveThumbnail } from "./WaveThumbnail";

const pieceTypes: PieceType[] = ["pawn", "rook", "spy", "king"];
const values: Coefficient[] = [1, -1];

function PieceLegend({ player, pieceType }: { player: Player; pieceType: PieceType }) {
  const piece: Piece = {
    id: `${player}-${pieceType}-control-legend`,
    owner: player,
    type: pieceType,
    position: { x: 0, y: 0 },
    unstable: false,
  };
  return (
    <span className="piece-shape-legend" aria-hidden="true">
      <PieceShape piece={piece} selected={false} dragging={false} />
    </span>
  );
}

export function coefficientLabel(player: Player, value: Coefficient) {
  const fieldSign = player === "blue" ? -value : value;
  return fieldSign === 1 ? "+" : fieldSign === -1 ? "-" : "0";
}

interface ComponentControlsProps {
  state: GameState;
  locked?: boolean;
  onTune: (pieceType: PieceType, componentIndex: number, value: Coefficient) => void;
  onRandomize: () => void;
  onReset: () => void;
}

export function ComponentControls({ state, locked = false, onTune, onRandomize, onReset }: ComponentControlsProps) {
  const player = state.currentPlayer;
  const controlsAtDefaults = JSON.stringify(state.components[player]) === JSON.stringify(state.defaultComponents);
  return (
    <aside className="control-panel" aria-label={`${player} component controls`}>
      <div className="legend">
        <span><i className="swatch red" /> &gt; 0</span>
        <span><i className="swatch neutral" /> = 0</span>
        <span><i className="swatch blue" /> &lt; 0</span>
        <button
          type="button"
          className="randomize-controls"
          title="Randomize tuning"
          aria-label="Randomize tuning"
          disabled={locked || state.status !== "playing"}
          onClick={onRandomize}
        >
          <Dices size={18} />
        </button>
        <button
          type="button"
          className="reset-controls"
          title="Reset tuning to defaults"
          aria-label="Reset tuning to defaults"
          disabled={locked || state.status !== "playing" || controlsAtDefaults}
          onClick={onReset}
        >
          <RotateCcw size={17} />
        </button>
      </div>
      {pieceTypes.map((pieceType) => (
        <section className={`component-group components-${COMPONENT_COUNTS[pieceType]}`} key={pieceType}>
          <div className="piece-heading">
            <WaveThumbnail state={state} player={player} pieceType={pieceType} />
            <PieceLegend player={player} pieceType={pieceType} />
            <div className="piece-heading-copy">
              <strong>{pieceType.toUpperCase()}</strong>
              <span className="component-count">({COMPONENT_COUNTS[pieceType]} components)</span>
              <span className="strength">
                <span className="strength-label">Active </span>
                {getTuningLoad(state.components[player][pieceType])}/{TUNING_STRENGTH[pieceType]}
              </span>
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
                  }
                  onClick={() => onTune(pieceType, index, value)}
                  aria-pressed={coefficient === value}
                  aria-label={`${player} ${pieceType} component ${index + 1} ${coefficient === value ? "turn off" : `set to ${coefficientLabel(player, value)}`}`}
                >
                  {coefficientLabel(player, value)}
                </button>
              ))}
            </div>
          ))}
        </section>
      ))}
    </aside>
  );
}
