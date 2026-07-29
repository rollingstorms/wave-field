import { FastForward, Pause, Play, StepForward } from "lucide-react";
import { BOARD_SIZE } from "../game/constants";
import { getLegalMoves } from "../game/movement";
import type { GameState, Player } from "../game/types";
import { getUnstablePieces, isKingUnprotected } from "../game/victory";

type AiMode = "off" | "red" | "duel";

interface AiDuelPanelProps {
  state: GameState;
  field: number[][];
  aiMode: AiMode;
  duelRunning: boolean;
  speedMs: number;
  maxTurns: number;
  onSetAiMode: (mode: AiMode) => void;
  onToggleRunning: () => void;
  onStep: () => void;
  onSetSpeed: (value: number) => void;
  onSetMaxTurns: (value: number) => void;
}

const players: Player[] = ["blue", "red"];
const materialValues = { pawn: 2, rook: 4, spy: 3, king: 0 } as const;

function playerName(player: Player) {
  return player === "blue" ? "Blue" : "Red";
}

function playerMetrics(state: GameState, field: number[][], player: Player) {
  const pieces = state.pieces.filter((piece) => piece.owner === player);
  const material = pieces.reduce((sum, piece) => sum + materialValues[piece.type], 0);
  const mobility = pieces.reduce((sum, piece) => sum + getLegalMoves(piece.id, state, field).length, 0);
  const unstable = getUnstablePieces(player, state, field).filter((piece) => piece.type !== "king").length;
  const kingInCheck = isKingUnprotected(player, state, field);
  const territory = field.reduce((sum, row) => sum + row.filter((value) => player === "red" ? value > 0 : value < 0).length, 0);
  return { player, pieces: pieces.length, material, mobility, unstable, kingInCheck, territory };
}

export function AiDuelPanel({
  state,
  field,
  aiMode,
  duelRunning,
  speedMs,
  maxTurns,
  onSetAiMode,
  onToggleRunning,
  onStep,
  onSetSpeed,
  onSetMaxTurns,
}: AiDuelPanelProps) {
  const metrics = players.map((player) => playerMetrics(state, field, player));
  const capReached = state.status === "playing" && aiMode === "duel" && state.turnNumber >= maxTurns;
  const canStep = state.status === "playing" && (aiMode === "duel" || (aiMode === "red" && state.currentPlayer === "red"));
  const boardSquares = BOARD_SIZE * BOARD_SIZE;

  return (
    <section className="ai-duel-panel" aria-labelledby="ai-duel-title">
      <header>
        <div>
          <h2 id="ai-duel-title">AI Arena</h2>
          <span>{aiMode === "duel" ? "Blue AI vs Red AI" : aiMode === "red" ? "Manual Blue vs Red AI" : "Manual play"}</span>
        </div>
        <select value={aiMode} onChange={(event) => onSetAiMode(event.currentTarget.value as AiMode)} aria-label="AI mode">
          <option value="duel">AI duel</option>
          <option value="red">Red AI only</option>
          <option value="off">Manual</option>
        </select>
      </header>

      <div className="duel-controls">
        <button type="button" className={duelRunning ? "active" : ""} onClick={onToggleRunning} disabled={aiMode !== "duel" || state.status !== "playing" || capReached}>
          {duelRunning ? <Pause size={17} /> : <Play size={17} />}
          {duelRunning ? "Pause" : "Run"}
        </button>
        <button type="button" onClick={onStep} disabled={!canStep}>
          <StepForward size={17} />
          Step
        </button>
      </div>

      <label className="range-control">
        <span><FastForward size={16} /> Speed {speedMs}ms</span>
        <input type="range" min="100" max="1400" step="50" value={speedMs} onChange={(event) => onSetSpeed(Number(event.currentTarget.value))} />
      </label>
      <label className="range-control">
        <span>Turn cap {maxTurns}</span>
        <input type="range" min="10" max="300" step="5" value={maxTurns} onChange={(event) => onSetMaxTurns(Number(event.currentTarget.value))} />
      </label>

      <div className="ai-metrics" aria-label="AI duel metrics">
        {metrics.map((metric) => (
          <article key={metric.player} className={metric.player}>
            <h3>{playerName(metric.player)}</h3>
            <b>{metric.kingInCheck ? "CHECK" : `${metric.territory}/${boardSquares}`}</b>
            <span>territory</span>
            <dl>
              <div><dt>Material</dt><dd>{metric.material}</dd></div>
              <div><dt>Mobility</dt><dd>{metric.mobility}</dd></div>
              <div><dt>Pieces</dt><dd>{metric.pieces}</dd></div>
              <div><dt>Unstable</dt><dd>{metric.unstable}</dd></div>
            </dl>
          </article>
        ))}
      </div>
      {capReached && <p className="duel-cap">Turn cap reached.</p>}
    </section>
  );
}
