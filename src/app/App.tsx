import { useEffect, useMemo, useReducer, useState } from "react";
import { Board } from "../components/Board";
import { ComponentControls } from "../components/ComponentControls";
import { DebugPanel } from "../components/DebugPanel";
import { HistoryRoll } from "../components/HistoryRoll";
import { RulesPage } from "../components/RulesPage";
import { TurnStatus } from "../components/TurnStatus";
import { WaveEditor } from "../components/WaveEditor";
import { cloneDefinitions, DEFAULT_DEFINITIONS } from "../field/componentDefinitions";
import { ALL_ENERGY_CHANNELS } from "../field/cmykEnergy";
import type { EnergyChannelState } from "../field/cmykEnergy";
import { evaluateField, evaluateTypeFields } from "../field/evaluateField";
import { createInitialState } from "../game/initialState";
import { gameReducer } from "../game/reducer";
import type { BasisDefinition, PieceType, Position } from "../game/types";

export function App() {
  const [state, dispatch] = useReducer(gameReducer, undefined, createInitialState);
  const [developerMode, setDeveloperMode] = useState(false);
  const [highContrast, setHighContrast] = useState(false);
  const [showTypeSums, setShowTypeSums] = useState(false);
  const [energyView, setEnergyView] = useState(false);
  const [energyChannels, setEnergyChannels] = useState<EnergyChannelState>({ ...ALL_ENERGY_CHANNELS });
  const [showRules, setShowRules] = useState(false);
  const [aiEnabled, setAiEnabled] = useState(true);
  const [aiThinking, setAiThinking] = useState(false);
  const [hintSearching, setHintSearching] = useState(false);
  const [editorSelection, setEditorSelection] = useState<{ pieceType: PieceType; componentIndex: number }>({ pieceType: "rook", componentIndex: 0 });
  const field = useMemo(() => evaluateField(state), [state]);
  const typeFields = useMemo(() => evaluateTypeFields(state), [state]);
  const aiTurn = aiEnabled && !energyView && state.currentPlayer === "red" && state.status === "playing";

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

  function requestHint() {
    if (hintSearching) return;
    setHintSearching(true);
    globalThis.setTimeout(() => {
      dispatch({ type: "hint" });
      setHintSearching(false);
    }, 40);
  }

  if (showRules) {
    return <RulesPage onBack={() => setShowRules(false)} />;
  }

  return (
    <main className={`app ${highContrast ? "high-contrast" : ""}`}>
      <TurnStatus
        state={state}
        developerMode={developerMode}
        highContrast={highContrast}
        showTypeSums={showTypeSums}
        energyView={energyView}
        aiEnabled={aiEnabled}
        aiThinking={aiThinking}
        onUndo={undo}
        onRestart={() => dispatch({ type: "restart", keepDefinitions: true })}
        onToggleDeveloper={() => setDeveloperMode((value) => !value)}
        onToggleContrast={() => setHighContrast((value) => !value)}
        onToggleTypeSums={() => {
          setEnergyView(false);
          setShowTypeSums((value) => !value);
        }}
        onToggleEnergyView={() => {
          setShowTypeSums(false);
          setEnergyView((value) => !value);
        }}
        onToggleAi={() => setAiEnabled((value) => !value)}
        onShowRules={() => setShowRules(true)}
      />
      <div className="play-area">
        <Board
          state={state}
          field={field}
          typeFields={typeFields}
          highContrast={highContrast}
          showTypeSums={showTypeSums}
          energyView={energyView}
          energyChannels={energyChannels}
          locked={aiTurn || energyView}
          onSelect={(pieceId) => dispatch({ type: "select", pieceId })}
          onMove={(pieceId: string, destination: Position) => dispatch({ type: "move", pieceId, destination })}
          onResign={() => dispatch({ type: "resign" })}
          onHint={requestHint}
          hintSearching={hintSearching}
          onToggleEnergyChannel={(pieceType) => setEnergyChannels((channels) => ({ ...channels, [pieceType]: !channels[pieceType] }))}
        />
        <div className="config-stack">
          <ComponentControls
            state={state}
            locked={aiTurn}
            onTune={(pieceType, componentIndex, value) => dispatch({ type: "tune", pieceType, componentIndex, value })}
            onRandomize={() => dispatch({ type: "randomize-tuning" })}
          />
          <HistoryRoll state={state} />
        </div>
      </div>
      {state.status !== "playing" && <div className="win-banner">{state.status === "red-won" ? "Red wins" : "Blue wins"}</div>}
      {developerMode && (
        <div className="developer">
          <DebugPanel
            state={state}
            field={field}
            onUpdateDefault={(pieceType, componentIndex, value) => dispatch({ type: "update-default-component", pieceType, componentIndex, value })}
            onResetDefaults={() => dispatch({ type: "reset-default-components" })}
            onRestart={() => dispatch({ type: "restart", keepDefinitions: true })}
          />
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
