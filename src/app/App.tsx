import { useEffect, useMemo, useReducer, useState } from "react";
import { Board } from "../components/Board";
import { ComponentControls } from "../components/ComponentControls";
import { DebugPanel } from "../components/DebugPanel";
import { TurnStatus } from "../components/TurnStatus";
import { WaveEditor } from "../components/WaveEditor";
import { cloneDefinitions, DEFAULT_DEFINITIONS } from "../field/componentDefinitions";
import { evaluateField, evaluateTypeFields } from "../field/evaluateField";
import { createInitialState } from "../game/initialState";
import { gameReducer } from "../game/reducer";
import type { BasisDefinition, PieceType, Position } from "../game/types";

export function App() {
  const [state, dispatch] = useReducer(gameReducer, undefined, createInitialState);
  const [developerMode, setDeveloperMode] = useState(false);
  const [highContrast, setHighContrast] = useState(false);
  const [showTypeSums, setShowTypeSums] = useState(false);
  const [aiEnabled, setAiEnabled] = useState(true);
  const [aiThinking, setAiThinking] = useState(false);
  const [editorSelection, setEditorSelection] = useState<{ pieceType: PieceType; componentIndex: number }>({ pieceType: "rook", componentIndex: 0 });
  const field = useMemo(() => evaluateField(state), [state]);
  const typeFields = useMemo(() => evaluateTypeFields(state), [state]);
  const aiTurn = aiEnabled && state.currentPlayer === "red" && state.status === "playing";

  useEffect(() => {
    if (!aiTurn) {
      setAiThinking(false);
      return;
    }
    setAiThinking(true);
    const timer = globalThis.setTimeout(() => dispatch({ type: "ai-turn" }), 450);
    return () => globalThis.clearTimeout(timer);
  }, [aiTurn, state.turnNumber]);

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

  function undo() {
    if (aiEnabled && state.history.at(-1)?.currentPlayer === "red") {
      setAiEnabled(false);
    }
    dispatch({ type: "undo" });
  }

  return (
    <main className={`app ${highContrast ? "high-contrast" : ""}`}>
      <TurnStatus
        state={state}
        developerMode={developerMode}
        highContrast={highContrast}
        showTypeSums={showTypeSums}
        aiEnabled={aiEnabled}
        aiThinking={aiThinking}
        onUndo={undo}
        onRestart={() => dispatch({ type: "restart", keepDefinitions: true })}
        onToggleDeveloper={() => setDeveloperMode((value) => !value)}
        onToggleContrast={() => setHighContrast((value) => !value)}
        onToggleTypeSums={() => setShowTypeSums((value) => !value)}
        onToggleAi={() => setAiEnabled((value) => !value)}
      />
      <div className="play-area">
        <Board
          state={state}
          field={field}
          typeFields={typeFields}
          highContrast={highContrast}
          showTypeSums={showTypeSums}
          locked={aiTurn}
          onSelect={(pieceId) => dispatch({ type: "select", pieceId })}
          onMove={(pieceId: string, destination: Position) => dispatch({ type: "move", pieceId, destination })}
        />
        <ComponentControls
          state={state}
          locked={aiTurn}
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
