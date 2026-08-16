import { FastForward, Pause, Play, StepForward } from "lucide-react";
import { BOARD_SIZE } from "../game/constants";
import { getLegalMoves } from "../game/movement";
import { policyLabel } from "../game/neuralAi";
import type { AiPolicy } from "../game/neuralAi";
import type { GameState, Player } from "../game/types";
import { getUnstablePieces, isKingUnprotected } from "../game/victory";

interface AiDuelPanelProps {
  state: GameState;
  field: number[][];
  neuralEnabled: boolean;
  sidePolicies: Record<Player, AiPolicy | "human">;
  duelRunning: boolean;
  aiStatus: string | null;
  aiStats: Record<Player, { turns: number; tuneActions: number; lastTurnTunes: number }>;
  speedMs: number;
  maxTurns: number;
  onSetSidePolicy: (player: Player, policy: AiPolicy | "human") => void;
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
  neuralEnabled,
  sidePolicies,
  duelRunning,
  aiStatus,
  aiStats,
  speedMs,
  maxTurns,
  onSetSidePolicy,
  onToggleRunning,
  onStep,
  onSetSpeed,
  onSetMaxTurns,
}: AiDuelPanelProps) {
  const metrics = players.map((player) => playerMetrics(state, field, player));
  const bothAutomated = sidePolicies.blue !== "human" && sidePolicies.red !== "human";
  const capReached = state.status === "playing" && bothAutomated && state.turnNumber >= maxTurns;
  const currentPolicy = sidePolicies[state.currentPlayer];
  const canStep = state.status === "playing" && currentPolicy !== "human" && !capReached;
  const boardSquares = BOARD_SIZE * BOARD_SIZE;
  const modeLabel = bothAutomated
    ? "Watch AI self-play"
    : sidePolicies.blue === "human" && sidePolicies.red === "human"
      ? "Manual play"
      : "Human vs AI";

  return (
    <section className="ai-duel-panel" aria-labelledby="ai-duel-title">
      <header>
        <div>
          <h2 id="ai-duel-title">{neuralEnabled ? "Local AI Arena" : "AI Arena"}</h2>
          <span>{modeLabel}</span>
        </div>
      </header>

      <div className="side-policy-grid" aria-label="Side policies">
        {players.map((player) => (
          <label key={player}>
            <span>{playerName(player)}</span>
            <select
              value={sidePolicies[player]}
              onChange={(event) => onSetSidePolicy(player, event.currentTarget.value as AiPolicy | "human")}
            >
              <option value="human">Human</option>
              <option value="easy">Easy</option>
              <option value="heuristic">Heuristic</option>
              <option value="hard">Hard</option>
              {neuralEnabled && <option value="neural-residual">{policyLabel("neural-residual")}</option>}
              {neuralEnabled && <option value="neural-transformer">{policyLabel("neural-transformer")}</option>}
            </select>
          </label>
        ))}
      </div>

      <div className="duel-controls">
        <button type="button" className={duelRunning ? "active" : ""} onClick={onToggleRunning} disabled={!bothAutomated || state.status !== "playing" || capReached}>
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
      {aiStatus && <p className="ai-status">{aiStatus}</p>}

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
              <div><dt>AI turns</dt><dd>{aiStats[metric.player].turns}</dd></div>
              <div><dt>Tune actions</dt><dd>{aiStats[metric.player].tuneActions}</dd></div>
              <div><dt>Last tune</dt><dd>{aiStats[metric.player].lastTurnTunes}</dd></div>
            </dl>
          </article>
        ))}
      </div>
      {capReached && <p className="duel-cap">Turn cap reached.</p>}
    </section>
  );
}
