import { useEffect, useMemo, useReducer, useState } from "react";
import { AiDuelPanel } from "../components/AiDuelPanel";
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

const arenaEnabled = import.meta.env.DEV && import.meta.env.MODE === "arena";

export function App() {
  const [state, dispatch] = useReducer(gameReducer, undefined, createInitialState);
  const [duelSeed, setDuelSeed] = useState(() => Math.floor(Math.random() * 1_000_000_000));
  const [developerMode, setDeveloperMode] = useState(false);
  const [continuousField, setContinuousField] = useState(false);
  const [showTypeSums, setShowTypeSums] = useState(false);
  const [energyView, setEnergyView] = useState(false);
  const [energyChannels, setEnergyChannels] = useState<EnergyChannelState>({ ...ALL_ENERGY_CHANNELS });
  const [showRules, setShowRules] = useState(false);
  const [aiMode, setAiMode] = useState<"off" | "red" | "duel">(() => arenaEnabled ? "duel" : "red");
  const [duelRunning, setDuelRunning] = useState(false);
  const [duelSpeedMs, setDuelSpeedMs] = useState(450);
  const [duelMaxTurns, setDuelMaxTurns] = useState(80);
  const [aiThinking, setAiThinking] = useState(false);
  const [hintSearching, setHintSearching] = useState(false);
  const [editorSelection, setEditorSelection] = useState<{ pieceType: PieceType; componentIndex: number }>({ pieceType: "rook", componentIndex: 0 });
  const field = useMemo(() => evaluateField(state), [state]);
  const typeFields = useMemo(() => evaluateTypeFields(state), [state]);
  const capReached = arenaEnabled && aiMode === "duel" && state.turnNumber >= duelMaxTurns;
  const aiTurn = !energyView
    && state.status === "playing"
    && ((aiMode === "red" && state.currentPlayer === "red") || (arenaEnabled && aiMode === "duel" && duelRunning && !capReached));

  useEffect(() => {
    if (!aiTurn) {
      setAiThinking(false);
      return;
    }
    setAiThinking(true);
    const timer = globalThis.setTimeout(() => {
      setAiThinking(false);
      dispatch({ type: "ai-turn", player: state.currentPlayer, seed: duelSeed, variety: arenaEnabled && aiMode === "duel" ? 0.55 : 0 });
    }, arenaEnabled && aiMode === "duel" ? duelSpeedMs : 450);
    return () => globalThis.clearTimeout(timer);
  }, [aiMode, aiTurn, duelSeed, duelSpeedMs, state.currentPlayer, state.turnNumber]);

  useEffect(() => {
    if (state.status !== "playing" || capReached) setDuelRunning(false);
  }, [capReached, state.status]);

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
    if (aiMode !== "off" && state.history.at(-1)?.currentPlayer === "red") {
      setAiMode("off");
      setDuelRunning(false);
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

  function restartGame() {
    setDuelRunning(false);
    setAiThinking(false);
    setDuelSeed(Math.floor(Math.random() * 1_000_000_000));
    dispatch({ type: "restart", keepDefinitions: true });
  }

  if (showRules) {
    return <RulesPage onBack={() => setShowRules(false)} />;
  }

  return (
    <main className="app">
      <TurnStatus
        state={state}
        developerMode={developerMode}
        continuousField={continuousField}
        showTypeSums={showTypeSums}
        energyView={energyView}
        aiEnabled={aiMode !== "off"}
        aiThinking={aiThinking}
        onUndo={undo}
        onRestart={restartGame}
        onToggleDeveloper={() => setDeveloperMode((value) => !value)}
        onToggleContinuousField={() => {
          setEnergyView(false);
          setContinuousField((value) => !value);
        }}
        onToggleTypeSums={() => {
          setEnergyView(false);
          setShowTypeSums((value) => !value);
        }}
        onToggleEnergyView={() => {
          setShowTypeSums(false);
          setContinuousField(false);
          setEnergyView((value) => !value);
        }}
        onToggleAi={() => {
          setAiMode((value) => {
            if (!arenaEnabled) return value === "off" ? "red" : "off";
            return value === "off" ? "red" : value === "red" ? "duel" : "off";
          });
          setDuelRunning(false);
        }}
        onShowRules={() => setShowRules(true)}
      />
      <div className="play-area">
        <Board
          state={state}
          field={field}
          typeFields={typeFields}
          continuousField={continuousField}
          showTypeSums={showTypeSums}
          energyView={energyView}
          energyChannels={energyChannels}
          locked={aiTurn || energyView || (arenaEnabled && aiMode === "duel")}
          onSelect={(pieceId) => dispatch({ type: "select", pieceId })}
          onMove={(pieceId: string, destination: Position) => dispatch({ type: "move", pieceId, destination })}
          onResign={() => dispatch({ type: "resign" })}
          onHint={requestHint}
          hintSearching={hintSearching}
          onToggleEnergyChannel={(pieceType) => setEnergyChannels((channels) => ({ ...channels, [pieceType]: !channels[pieceType] }))}
        />
        <div className="config-stack">
          {arenaEnabled && (
            <AiDuelPanel
              state={state}
              field={field}
              aiMode={aiMode}
              duelRunning={duelRunning}
              speedMs={duelSpeedMs}
              maxTurns={duelMaxTurns}
              onSetAiMode={(mode) => {
                setAiMode(mode);
                setDuelRunning(false);
              }}
              onToggleRunning={() => setDuelRunning((value) => !value)}
              onStep={() => dispatch({ type: "ai-turn", player: state.currentPlayer, seed: duelSeed, variety: aiMode === "duel" ? 0.55 : 0 })}
              onSetSpeed={setDuelSpeedMs}
              onSetMaxTurns={setDuelMaxTurns}
            />
          )}
          <ComponentControls
            state={state}
            locked={aiTurn || (arenaEnabled && aiMode === "duel")}
            onTune={(pieceType, componentIndex, value) => dispatch({ type: "tune", pieceType, componentIndex, value })}
            onRandomize={() => dispatch({ type: "randomize-tuning" })}
            onReset={() => dispatch({ type: "reset-tuning" })}
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
            onUpdateWaveScale={(pieceType, scale, value) => dispatch({ type: "update-wave-scale", pieceType, scale, value })}
            onResetWaveScales={() => dispatch({ type: "reset-wave-scales" })}
            onUpdateHomeEnergy={(pieceType, value) => dispatch({ type: "update-home-energy", pieceType, value })}
            onResetHomeEnergy={() => dispatch({ type: "reset-home-energy" })}
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
