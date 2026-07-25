import { BOARD_SIZE } from "../game/constants";
import { contributionGrid, evaluateSignedPieceContribution } from "../field/evaluateField";
import { projectFieldValue } from "../field/projection";
import { getLegalMoves } from "../game/movement";
import type { GameState } from "../game/types";

interface DebugPanelProps {
  state: GameState;
  field: number[][];
}

export function DebugPanel({ state, field }: DebugPanelProps) {
  const values = field.flat();
  const globalBias = values.reduce((sum, value) => sum + value, 0);
  const energy = values.reduce((sum, value) => sum + value * value, 0);
  const symmetry = field.reduce((sum, row, y) => sum + row.reduce((inner, value, x) => {
    const mirror = field[BOARD_SIZE - 1 - y][BOARD_SIZE - 1 - x];
    return inner + Math.pow(value + mirror, 2);
  }, 0), 0);
  const counts = values.reduce((acc, value) => {
    acc[projectFieldValue(value)] += 1;
    return acc;
  }, { red: 0, neutral: 0, blue: 0 });

  return (
    <section className="debug-panel">
      <div className="metrics">
        <span>Bias {globalBias.toFixed(6)}</span>
        <span>Energy {energy.toFixed(3)}</span>
        <span>Symmetry {symmetry.toExponential(2)}</span>
        <span>R/N/B {counts.red}/{counts.neutral}/{counts.blue}</span>
      </div>
      <div className="field-table" aria-label="Raw field values">
        {field.map((row, y) => row.map((value, x) => <span key={`${x}-${y}`}>{value.toFixed(2)}</span>))}
      </div>
      <details>
        <summary>Pieces and mobility</summary>
        <div className="piece-list">
          {state.pieces.map((piece) => (
            <span key={piece.id}>
              {piece.id}: ({piece.position.x},{piece.position.y}) field {field[piece.position.y][piece.position.x].toFixed(3)} moves {getLegalMoves(piece.id, state, field).length}
            </span>
          ))}
        </div>
      </details>
      <details>
        <summary>Contribution at selected square</summary>
        <div className="piece-list">
          {state.pieces.map((piece) => (
            <span key={piece.id}>{piece.id}: center contribution {evaluateSignedPieceContribution(piece, piece.position, state).toFixed(3)}</span>
          ))}
        </div>
      </details>
      <details>
        <summary>Selected piece contribution grid</summary>
        <div className="field-table">
          {(state.pieces.find((piece) => piece.id === state.selectedPieceId) ? contributionGrid(state.pieces.find((piece) => piece.id === state.selectedPieceId)!, state) : field).map((row, y) =>
            row.map((value, x) => <span key={`${x}-${y}`}>{value.toFixed(2)}</span>),
          )}
        </div>
      </details>
    </section>
  );
}
