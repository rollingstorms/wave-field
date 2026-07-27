import { Bot, CircleHelp, Grid2X2, Palette, RotateCcw, Undo2, Wrench } from "lucide-react";
import type { GameState } from "../game/types";

interface TurnStatusProps {
  state: GameState;
  developerMode: boolean;
  continuousField: boolean;
  showTypeSums: boolean;
  energyView: boolean;
  aiEnabled: boolean;
  aiThinking: boolean;
  onUndo: () => void;
  onRestart: () => void;
  onToggleDeveloper: () => void;
  onToggleContinuousField: () => void;
  onToggleTypeSums: () => void;
  onToggleEnergyView: () => void;
  onToggleAi: () => void;
  onShowRules: () => void;
}

export function TurnStatus({ state, developerMode, continuousField, showTypeSums, energyView, aiEnabled, aiThinking, onUndo, onRestart, onToggleDeveloper, onToggleContinuousField, onToggleTypeSums, onToggleEnergyView, onToggleAi, onShowRules }: TurnStatusProps) {
  const winner = state.status === "red-won" ? "Red wins" : state.status === "blue-won" ? "Blue wins" : null;
  return (
    <header className="topbar">
      <div>
        <p className={`side-label ${state.currentPlayer}`}>{state.currentPlayer.toUpperCase()}</p>
        <h1>Wave Field</h1>
      </div>
      <div className="turn-block">
        <span>TURN {state.turnNumber}</span>
        <strong>{winner ?? (aiThinking ? "Red AI thinking..." : state.message)}</strong>
      </div>
      <div className="actions">
        <button title="How to play" aria-label="How to play" onClick={onShowRules}><CircleHelp size={18} /></button>
        <button title="Undo" aria-label="Undo" onClick={onUndo}><Undo2 size={18} /></button>
        <button title="Restart" aria-label="Restart" onClick={onRestart}><RotateCcw size={18} /></button>
        <button className={developerMode ? "active" : ""} title="Developer mode" aria-label="Developer mode" onClick={onToggleDeveloper}><Wrench size={18} /></button>
        <button
          className={showTypeSums ? "active" : ""}
          title="Piece-type field sums"
          aria-label="Piece-type field sums"
          aria-pressed={showTypeSums}
          onClick={onToggleTypeSums}
        >
          <Grid2X2 size={18} />
        </button>
        <button
          className={energyView ? "active" : ""}
          title="CMYK energy view"
          aria-label="CMYK energy view"
          aria-pressed={energyView}
          onClick={onToggleEnergyView}
        >
          <Palette size={18} />
        </button>
        <button
          className={aiEnabled ? "active" : ""}
          title={aiEnabled ? "Red AI enabled" : "Red AI disabled"}
          aria-label="Red AI opponent"
          aria-pressed={aiEnabled}
          onClick={onToggleAi}
        >
          <Bot size={18} />
        </button>
        <button
          className={`gradient-view-toggle ${continuousField ? "active" : ""}`}
          title="Continuous field shading"
          aria-label="Continuous field shading"
          aria-pressed={continuousField}
          onClick={onToggleContinuousField}
        >
          <span className="gradient-swatch" aria-hidden="true" />
        </button>
      </div>
    </header>
  );
}
