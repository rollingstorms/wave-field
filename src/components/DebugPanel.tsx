import { BOARD_SIZE, DEBUG_COMPONENT_COUNT_LIMITS } from "../game/constants";
import { contributionGrid, evaluateSignedPieceContribution } from "../field/evaluateField";
import { projectFieldValue } from "../field/projection";
import { getLegalMoves } from "../game/movement";
import { PIECE_TYPES, pieceName } from "../game/pieceLabels";
import { getUnstablePieces, isKingUnprotected } from "../game/victory";
import type { GameSnapshot, GameState } from "../game/types";
import type { Coefficient, HomeEnergy, PieceType, Player, WaveScales } from "../game/types";
import { DefaultComponentEditor } from "./DefaultComponentEditor";
import { HistoryRoll } from "./HistoryRoll";

interface DebugPanelProps {
  state: GameState;
  field: number[][];
  onUpdateDefault: (pieceType: PieceType, componentIndex: number, value: Coefficient) => void;
  onUpdateWaveScale: (pieceType: PieceType, scale: keyof WaveScales[PieceType], value: number) => void;
  onResetWaveScales: () => void;
  onUpdateHomeEnergy: (pieceType: PieceType, value: number) => void;
  onResetHomeEnergy: () => void;
  onSetComponentCount: (pieceType: PieceType, count: number) => void;
  onResetDefaults: () => void;
  onRestart: () => void;
}

const pieceTypes: PieceType[] = PIECE_TYPES;
const players: Player[] = ["blue", "red"];

interface PressureMetric {
  player: Player;
  legalMoves: number;
  unstable: number;
  kingInCheck: boolean;
}

function playerName(player: Player) {
  return player === "blue" ? "Blue" : "Red";
}

function pressureMetrics(snap: GameSnapshot): PressureMetric[] {
  const state = snap as GameState;
  const field = evaluateFieldForSnapshot(state);
  return players.map((player) => {
    const pieces = state.pieces.filter((piece) => piece.owner === player);
    const legalMoves = pieces.reduce((total, piece) => total + getLegalMoves(piece.id, state, field).length, 0);
    const unstable = getUnstablePieces(player, state, field).filter((piece) => piece.type !== "king").length;
    return { player, legalMoves, unstable, kingInCheck: isKingUnprotected(player, state, field) };
  });
}

function evaluateFieldForSnapshot(state: GameState) {
  return Array.from({ length: BOARD_SIZE }, (_, y) =>
    Array.from({ length: BOARD_SIZE }, (_, x) =>
      state.pieces.reduce((total, piece) => total + evaluateSignedPieceContribution(piece, { x, y }, state), 0),
    ),
  );
}

function pressureScore(metrics: PressureMetric[]) {
  const blue = metrics.find((metric) => metric.player === "blue")!;
  const red = metrics.find((metric) => metric.player === "red")!;
  return (blue.kingInCheck ? 80 : 0)
    - (red.kingInCheck ? 80 : 0)
    + blue.unstable * 20
    - red.unstable * 20
    + red.legalMoves
    - blue.legalMoves;
}

function WaveScaleEditor({ scales, onUpdate, onReset }: {
  scales: WaveScales;
  onUpdate: DebugPanelProps["onUpdateWaveScale"];
  onReset: () => void;
}) {
  return (
    <section className="wave-scale-editor" aria-labelledby="wave-scale-title">
      <div className="debug-section-heading">
        <h2 id="wave-scale-title">Wave Scales</h2>
        <button type="button" className="secondary" onClick={onReset}>Reset</button>
      </div>
      <div className="wave-scale-grid">
        <strong>Piece</strong>
        <strong>Friendly</strong>
        <strong>Hostile</strong>
        {pieceTypes.map((pieceType) => (
          <div className="wave-scale-row" key={pieceType}>
            <span>{pieceName(pieceType).toUpperCase()}</span>
            <input
              type="number"
              min="0"
              max="4"
              step="0.25"
              value={scales[pieceType].friendly}
              onChange={(event) => onUpdate(pieceType, "friendly", Number(event.currentTarget.value))}
              aria-label={`${pieceName(pieceType)} friendly scale`}
            />
            <input
              type="number"
              min="0"
              max="4"
              step="0.25"
              value={scales[pieceType].hostile}
              onChange={(event) => onUpdate(pieceType, "hostile", Number(event.currentTarget.value))}
              aria-label={`${pieceName(pieceType)} hostile scale`}
            />
          </div>
        ))}
      </div>
    </section>
  );
}

function HomeEnergyEditor({ homeEnergy, onUpdate, onReset }: {
  homeEnergy: HomeEnergy;
  onUpdate: DebugPanelProps["onUpdateHomeEnergy"];
  onReset: () => void;
}) {
  return (
    <section className="home-energy-editor" aria-labelledby="home-energy-title">
      <div className="debug-section-heading">
        <h2 id="home-energy-title">Home Energy</h2>
        <button type="button" className="secondary" onClick={onReset}>Reset</button>
      </div>
      <div className="home-energy-grid">
        <strong>Piece</strong>
        <strong>Home</strong>
        {pieceTypes.map((pieceType) => (
          <div className="home-energy-row" key={pieceType}>
            <span>{pieceName(pieceType).toUpperCase()}</span>
            <input
              type="number"
              min="-4"
              max="4"
              step="0.25"
              value={homeEnergy[pieceType]}
              onChange={(event) => onUpdate(pieceType, Number(event.currentTarget.value))}
              aria-label={`${pieceName(pieceType)} home energy`}
            />
          </div>
        ))}
      </div>
    </section>
  );
}

function PatternCountEditor({ state, onUpdate }: {
  state: GameState;
  onUpdate: DebugPanelProps["onSetComponentCount"];
}) {
  return (
    <section className="pattern-count-editor" aria-labelledby="pattern-count-title">
      <div className="debug-section-heading">
        <h2 id="pattern-count-title">Pattern Slots</h2>
      </div>
      <div className="home-energy-grid">
        <strong>Piece</strong>
        <strong>Slots</strong>
        {pieceTypes.map((pieceType) => (
          <div className="home-energy-row" key={pieceType}>
            <span>{pieceName(pieceType).toUpperCase()}</span>
            <div className="slot-stepper">
              <button
                aria-label={`Remove ${pieceName(pieceType)} pattern slot`}
                disabled={state.components[state.currentPlayer][pieceType].length <= 1}
                onClick={() => onUpdate(pieceType, state.components[state.currentPlayer][pieceType].length - 1)}
              >
                -
              </button>
              <span>{state.components[state.currentPlayer][pieceType].length}</span>
              <button
                aria-label={`Add ${pieceName(pieceType)} pattern slot`}
                disabled={state.components[state.currentPlayer][pieceType].length >= DEBUG_COMPONENT_COUNT_LIMITS[pieceType]}
                onClick={() => onUpdate(pieceType, state.components[state.currentPlayer][pieceType].length + 1)}
              >
                +
              </button>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function PressurePanel({ state }: { state: GameState }) {
  const current = pressureMetrics(state);
  const score = pressureScore(current);
  const trend = [...state.history.slice(-18), state].map((snap, index) => ({
    index,
    score: pressureScore(pressureMetrics(snap)),
  }));
  const maxMagnitude = Math.max(1, ...trend.map((entry) => Math.abs(entry.score)));

  return (
    <section className="pressure-panel" aria-labelledby="pressure-title">
      <div className="debug-section-heading">
        <h2 id="pressure-title">Pressure</h2>
        <span>Red {score >= 0 ? "+" : ""}{score.toFixed(0)}</span>
      </div>
      <div className="pressure-metrics">
        {current.map((metric) => (
          <span key={metric.player}>
            <b>{playerName(metric.player)}</b>
            <small>moves {metric.legalMoves}</small>
            <small>unstable {metric.unstable}</small>
            <small>{metric.kingInCheck ? "big hat in check" : "big hat safe"}</small>
          </span>
        ))}
      </div>
      <div className="pressure-trend" aria-label="Pressure trend over recent moves">
        {trend.map((entry) => (
          <i
            key={entry.index}
            className={entry.score >= 0 ? "red" : "blue"}
            style={{ height: `${10 + (Math.abs(entry.score) / maxMagnitude) * 42}px` }}
            title={`Pressure ${entry.score.toFixed(0)}`}
          />
        ))}
      </div>
    </section>
  );
}

export function DebugPanel({ state, field, onUpdateDefault, onUpdateWaveScale, onResetWaveScales, onUpdateHomeEnergy, onResetHomeEnergy, onSetComponentCount, onResetDefaults, onRestart }: DebugPanelProps) {
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
      <DefaultComponentEditor
        defaults={state.defaultComponents}
        onUpdate={onUpdateDefault}
        onReset={onResetDefaults}
        onRestart={onRestart}
      />
      <WaveScaleEditor scales={state.waveScales} onUpdate={onUpdateWaveScale} onReset={onResetWaveScales} />
      <HomeEnergyEditor homeEnergy={state.homeEnergy} onUpdate={onUpdateHomeEnergy} onReset={onResetHomeEnergy} />
      <PatternCountEditor state={state} onUpdate={onSetComponentCount} />
      <PressurePanel state={state} />
      <HistoryRoll state={state} />
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
