import { RotateCcw, Undo2, Wrench } from "lucide-react";
import type { GameState } from "../game/types";

interface TurnStatusProps {
  state: GameState;
  developerMode: boolean;
  highContrast: boolean;
  onUndo: () => void;
  onRestart: () => void;
  onToggleDeveloper: () => void;
  onToggleContrast: () => void;
}

export function TurnStatus({ state, developerMode, highContrast, onUndo, onRestart, onToggleDeveloper, onToggleContrast }: TurnStatusProps) {
  const winner = state.status === "red-won" ? "Red wins" : state.status === "blue-won" ? "Blue wins" : null;
  return (
    <header className="topbar">
      <div>
        <p className={`side-label ${state.currentPlayer}`}>{state.currentPlayer.toUpperCase()}</p>
        <h1>Wave Field</h1>
      </div>
      <div className="turn-block">
        <span>TURN {state.turnNumber}</span>
        <strong>{winner ?? state.message}</strong>
      </div>
      <div className="actions">
        <button title="Undo" aria-label="Undo" onClick={onUndo}><Undo2 size={18} /></button>
        <button title="Restart" aria-label="Restart" onClick={onRestart}><RotateCcw size={18} /></button>
        <button className={developerMode ? "active" : ""} title="Developer mode" aria-label="Developer mode" onClick={onToggleDeveloper}><Wrench size={18} /></button>
        <button className={highContrast ? "active" : ""} onClick={onToggleContrast}>A11Y</button>
      </div>
    </header>
  );
}
