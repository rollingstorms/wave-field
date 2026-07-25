import { useMemo, useReducer, useState } from "react";
import { Board } from "../components/Board";
import { ComponentControls } from "../components/ComponentControls";
import { DebugPanel } from "../components/DebugPanel";
import { TurnStatus } from "../components/TurnStatus";
import { WaveEditor } from "../components/WaveEditor";
import { cloneDefinitions, DEFAULT_DEFINITIONS } from "../field/componentDefinitions";
import { evaluateField } from "../field/evaluateField";
import { createInitialState } from "../game/initialState";
import { gameReducer } from "../game/reducer";
import type { BasisDefinition, PieceType, Position } from "../game/types";

export function App() {
  const [state, dispatch] = useReducer(gameReducer, undefined, createInitialState);
  const [developerMode, setDeveloperMode] = useState(false);
  const [highContrast, setHighContrast] = useState(false);
  const [editorSelection, setEditorSelection] = useState<{ pieceType: PieceType; componentIndex: number }>({ pieceType: "rook", componentIndex: 0 });
  const field = useMemo(() => evaluateField(state), [state]);

  function updateDefinition(definition: BasisDefinition) {
    dispatch({ type: "update-definition", ...editorSelection, definition });
  }

  function resetSelectedDefinition() {
    dispatch({
      type: "update-definition",
      ...editorSelection,
      definition: structuredClone(DEFAULT_DEFINITIONS[editorSelection.pieceType][editorSelection.componentIndex]),
    });
  }

  return (
    <main className={`app ${highContrast ? "high-contrast" : ""}`}>
      <TurnStatus
        state={state}
        developerMode={developerMode}
        highContrast={highContrast}
        onUndo={() => dispatch({ type: "undo" })}
        onRestart={() => dispatch({ type: "restart", keepDefinitions: true })}
        onToggleDeveloper={() => setDeveloperMode((value) => !value)}
        onToggleContrast={() => setHighContrast((value) => !value)}
      />
      <div className="play-area">
        <Board
          state={state}
          field={field}
          highContrast={highContrast}
          onSelect={(pieceId) => dispatch({ type: "select", pieceId })}
          onMove={(pieceId: string, destination: Position) => dispatch({ type: "move", pieceId, destination })}
        />
        <ComponentControls
          state={state}
          onTune={(pieceType, componentIndex, value) => dispatch({ type: "tune", pieceType, componentIndex, value })}
        />
      </div>
      {state.status !== "playing" && <div className="win-banner">{state.status === "red-won" ? "Red wins" : "Blue wins"}</div>}
      {developerMode && (
        <div className="developer">
          <DebugPanel state={state} field={field} />
          <WaveEditor
            definitions={state.definitions}
            selected={editorSelection}
            onSelect={setEditorSelection}
            onUpdate={updateDefinition}
            onResetSelected={resetSelectedDefinition}
            onResetAll={() => dispatch({ type: "reset-definitions" })}
            onImport={(payload) => dispatch({ type: "import-definitions", payload })}
          />
          <button className="secondary" onClick={() => dispatch({ type: "restart", keepDefinitions: false })}>Restart and reset definitions</button>
        </div>
      )}
    </main>
  );
}
