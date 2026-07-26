import { RotateCcw } from "lucide-react";
import { COMPONENT_COUNTS, PIECE_STRENGTH } from "../game/constants";
import { isTuningWithinStrength } from "../game/tuning";
import type { Coefficient, PieceType, PlayerComponents } from "../game/types";

const pieceTypes: PieceType[] = ["pawn", "rook", "spy", "king"];
const values: Coefficient[] = [1, 0, -1];
const coefficientLabel = (value: Coefficient) => value === 1 ? "+" : value === -1 ? "-" : "0";

interface DefaultComponentEditorProps {
  defaults: PlayerComponents;
  onUpdate: (pieceType: PieceType, componentIndex: number, value: Coefficient) => void;
  onReset: () => void;
  onRestart: () => void;
}

export function DefaultComponentEditor({ defaults, onUpdate, onReset, onRestart }: DefaultComponentEditorProps) {
  function canSet(pieceType: PieceType, componentIndex: number, value: Coefficient) {
    const coefficients = [...defaults[pieceType]];
    coefficients[componentIndex] = value;
    return isTuningWithinStrength(pieceType, coefficients);
  }

  return (
    <section className="default-component-editor" aria-labelledby="default-controls-title">
      <div className="default-editor-heading">
        <div>
          <h2 id="default-controls-title">Default Controls</h2>
          <p>Applied to both players when the game restarts.</p>
        </div>
        <button type="button" className="icon-button" title="Reset control defaults" aria-label="Reset control defaults" onClick={onReset}>
          <RotateCcw size={17} />
        </button>
      </div>
      <div className="default-piece-list">
        {pieceTypes.map((pieceType) => (
          <div className="default-piece-row" key={pieceType}>
            <span className="default-piece-name">
              <strong>{pieceType.toUpperCase()}</strong>
              <small>Strength {PIECE_STRENGTH[pieceType]}</small>
            </span>
            <div className={`default-component-list components-${COMPONENT_COUNTS[pieceType]}`}>
              {defaults[pieceType].map((coefficient, componentIndex) => (
                <div className="default-component" key={`${pieceType}-${componentIndex}`}>
                  <span>C{componentIndex + 1}</span>
                  <div className="default-segmented">
                    {values.map((value) => (
                      <button
                        type="button"
                        key={value}
                        className={coefficient === value ? "active" : ""}
                        disabled={coefficient === value || !canSet(pieceType, componentIndex, value)}
                        aria-pressed={coefficient === value}
                        aria-label={`Default ${pieceType} component ${componentIndex + 1} set to ${coefficientLabel(value)}`}
                        onClick={() => onUpdate(pieceType, componentIndex, value)}
                      >
                        {coefficientLabel(value)}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
      <button type="button" className="secondary restart-with-defaults" onClick={onRestart}>Restart with defaults</button>
    </section>
  );
}
