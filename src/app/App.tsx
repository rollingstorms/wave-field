import { useEffect, useMemo, useReducer, useState } from "react";
import { AiDuelPanel } from "../components/AiDuelPanel";
import { Board } from "../components/Board";
import { ComponentControls } from "../components/ComponentControls";
import { DebugPanel } from "../components/DebugPanel";
import { HistoryRoll } from "../components/HistoryRoll";
import { RulesPage } from "../components/RulesPage";
import { GameActions, TurnStatus } from "../components/TurnStatus";
import { WaveEditor } from "../components/WaveEditor";
import { definitionForSlot, TRAINING_COMPONENTS } from "../field/componentDefinitions";
import { ALL_ENERGY_CHANNELS } from "../field/cmykEnergy";
import type { EnergyChannelState } from "../field/cmykEnergy";
import { evaluateField, evaluateTypeFields } from "../field/evaluateField";
import { playEasyTurn, playHeuristicTurn } from "../game/ai";
import { BOARD_SIZE } from "../game/constants";
import { createInitialState } from "../game/initialState";
import { isNeuralPolicy, policyLabel, requestNeuralTurn } from "../game/neuralAi";
import type { AiPolicy } from "../game/neuralAi";
import { gameReducer } from "../game/reducer";
import type { CSSProperties } from "react";
import type { BasisDefinition, PieceType, Player, PlayerComponents, Position } from "../game/types";

const routePath = globalThis.location?.pathname.replace(/\/$/, "") ?? "";
const localNeuralArenaEnabled = import.meta.env.DEV
  && (routePath.endsWith("/local-arena") || import.meta.env.MODE === "arena");
const arenaEnabled = routePath.endsWith("/arena") || localNeuralArenaEnabled;
type SidePolicy = AiPolicy | "human";
type AiStats = Record<Player, { turns: number; tuneActions: number; lastTurnTunes: number }>;
const pieceTypes: PieceType[] = ["pawn", "rook", "spy", "king"];
const createArenaInitialState = () => localNeuralArenaEnabled
  ? createInitialState(TRAINING_COMPONENTS)
  : createInitialState();

const emptyAiStats = (): AiStats => ({
  blue: { turns: 0, tuneActions: 0, lastTurnTunes: 0 },
  red: { turns: 0, tuneActions: 0, lastTurnTunes: 0 },
});

function componentChangeCount(before: PlayerComponents, after: PlayerComponents): number {
  return pieceTypes.reduce((total, pieceType) => (
    total + before[pieceType].filter((coefficient, index) => coefficient !== after[pieceType][index]).length
  ), 0);
}

export function App() {
  const [state, dispatch] = useReducer(gameReducer, undefined, createArenaInitialState);
  const [duelSeed, setDuelSeed] = useState(() => Math.floor(Math.random() * 1_000_000_000));
  const [developerMode, setDeveloperMode] = useState(false);
  const [continuousField, setContinuousField] = useState(false);
  const [showTypeSums, setShowTypeSums] = useState(false);
  const [energyView, setEnergyView] = useState(false);
  const [energyChannels, setEnergyChannels] = useState<EnergyChannelState>({ ...ALL_ENERGY_CHANNELS });
  const [showRules, setShowRules] = useState(false);
  const [sidePolicies, setSidePolicies] = useState<Record<Player, SidePolicy>>(() => arenaEnabled
    ? { blue: "easy", red: "easy" }
    : { blue: "human", red: "easy" });
  const [duelRunning, setDuelRunning] = useState(false);
  const [duelSpeedMs, setDuelSpeedMs] = useState(450);
  const [duelMaxTurns, setDuelMaxTurns] = useState(80);
  const [aiThinking, setAiThinking] = useState(false);
  const [aiStatus, setAiStatus] = useState<string | null>(null);
  const [aiStats, setAiStats] = useState<AiStats>(emptyAiStats);
  const [hintSearching, setHintSearching] = useState(false);
  const [editorSelection, setEditorSelection] = useState<{ pieceType: PieceType; componentIndex: number }>({ pieceType: "rook", componentIndex: 0 });
  const field = useMemo(() => evaluateField(state), [state]);
  const typeFields = useMemo(() => evaluateTypeFields(state), [state]);
  const bothAutomated = sidePolicies.blue !== "human" && sidePolicies.red !== "human";
  const capReached = arenaEnabled && bothAutomated && state.turnNumber >= duelMaxTurns;
  const currentPolicy = sidePolicies[state.currentPlayer];
  const aiTurn = !energyView
    && state.status === "playing"
    && currentPolicy !== "human"
    && !capReached
    && (!bothAutomated || duelRunning);

  function setSidePolicy(player: Player, policy: SidePolicy) {
    setSidePolicies((policies) => ({ ...policies, [player]: policy }));
    setDuelRunning(false);
    setAiStatus(null);
  }

  async function playAutomatedTurn(policy: AiPolicy, immediate = false) {
    if (state.status !== "playing" || capReached || energyView) return;
    setAiThinking(true);
    setAiStatus(`${policyLabel(policy)} thinking`);
    const run = async () => {
      if (policy === "easy" || policy === "heuristic") {
        const player = state.currentPlayer;
        const variety = policy === "heuristic" && bothAutomated ? 0.55 : 0;
        const preview = policy === "easy"
          ? playEasyTurn(state, player, { seed: duelSeed, variety })
          : playHeuristicTurn(state, player, { seed: duelSeed, variety });
        const tuneCount = componentChangeCount(state.components[player], preview.components[player]);
        setAiStats((stats) => ({
          ...stats,
          [player]: {
            turns: stats[player].turns + 1,
            tuneActions: stats[player].tuneActions + tuneCount,
            lastTurnTunes: tuneCount,
          },
        }));
        if (policy === "heuristic") {
          dispatch({ type: "ai-turn", player, seed: duelSeed, variety });
        } else {
          dispatch({ type: "replace-state", state: preview });
        }
        return;
      }
      if (!isNeuralPolicy(policy)) return;
      if (!localNeuralArenaEnabled) {
        setAiStatus("Neural models are available only in the local arena");
        return;
      }
      const response = await requestNeuralTurn(state, policy);
      const actions = response.actions;
      const tuneCount = actions.filter((action) => action.type === "tune").length;
      setAiStats((stats) => ({
        ...stats,
        [state.currentPlayer]: {
          turns: stats[state.currentPlayer].turns + 1,
          tuneActions: stats[state.currentPlayer].tuneActions + tuneCount,
          lastTurnTunes: tuneCount,
        },
      }));
      for (const action of actions) {
        if (action.type === "tune") {
          dispatch({
            type: "tune",
            pieceType: action.pieceType,
            componentIndex: action.componentIndex,
            value: action.value,
          });
        } else {
          dispatch({ type: "move", pieceId: action.pieceId, destination: action.destination });
        }
      }
      if (response.terminalState) {
        dispatch({ type: "replace-state", state: response.terminalState });
      }
    };
    try {
      if (!immediate) await new Promise((resolve) => globalThis.setTimeout(resolve, bothAutomated ? duelSpeedMs : 450));
      await run();
      setAiStatus(null);
    } catch (error) {
      setDuelRunning(false);
      setAiStatus(error instanceof Error ? error.message : "Neural model request failed");
    } finally {
      setAiThinking(false);
    }
  }

  useEffect(() => {
    if (!aiTurn) {
      setAiThinking(false);
      return;
    }
    let cancelled = false;
    const automatedPolicy = currentPolicy;
    const timer = globalThis.setTimeout(() => {
      if (!cancelled) void playAutomatedTurn(automatedPolicy, true);
    }, bothAutomated ? duelSpeedMs : 450);
    return () => {
      cancelled = true;
      globalThis.clearTimeout(timer);
    };
  }, [aiTurn, bothAutomated, currentPolicy, duelSeed, duelSpeedMs, state.currentPlayer, state.turnNumber]);

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
      definition: definitionForSlot(editorSelection.pieceType, editorSelection.componentIndex),
    });
  }

  function setComponentCount(pieceType: PieceType, count: number) {
    dispatch({ type: "set-component-count", pieceType, count });
    if (editorSelection.pieceType === pieceType && editorSelection.componentIndex >= count) {
      setEditorSelection({ pieceType, componentIndex: Math.max(0, count - 1) });
    }
  }

  function undo() {
    setDuelRunning(false);
    dispatch({ type: "undo" });
  }

  function requestHint(focusedPieceId: string | null = state.selectedPieceId) {
    if (hintSearching) return;
    setHintSearching(true);
    globalThis.setTimeout(() => {
      dispatch({ type: "hint", focusedPieceId });
      setHintSearching(false);
    }, 40);
  }

  function restartGame() {
    setDuelRunning(false);
    setAiThinking(false);
    setAiStatus(null);
    setAiStats(emptyAiStats());
    setDuelSeed(Math.floor(Math.random() * 1_000_000_000));
    dispatch({ type: "restart", keepDefinitions: true });
  }

  if (showRules) {
    return <RulesPage onBack={() => setShowRules(false)} />;
  }

  return (
    <main className="app" style={{ "--board-size": BOARD_SIZE } as CSSProperties}>
      <TurnStatus
        state={state}
        aiThinking={aiThinking}
        actions={(
          <GameActions
            developerMode={developerMode}
            continuousField={continuousField}
            showTypeSums={showTypeSums}
            energyView={energyView}
            aiEnabled={sidePolicies.blue !== "human" || sidePolicies.red !== "human"}
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
              setSidePolicies((policies) => (
                policies.blue === "human" && policies.red === "human"
                  ? { blue: "human", red: "heuristic" }
                  : { blue: "human", red: "human" }
              ));
              setDuelRunning(false);
            }}
            onShowRules={() => setShowRules(true)}
          />
        )}
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
          locked={aiTurn || energyView || currentPolicy !== "human"}
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
              neuralEnabled={localNeuralArenaEnabled}
              sidePolicies={sidePolicies}
              duelRunning={duelRunning}
              aiStatus={aiStatus}
              aiStats={aiStats}
              speedMs={duelSpeedMs}
              maxTurns={duelMaxTurns}
              onSetSidePolicy={setSidePolicy}
              onToggleRunning={() => setDuelRunning((value) => !value)}
              onStep={() => {
                if (currentPolicy !== "human") void playAutomatedTurn(currentPolicy, true);
              }}
              onSetSpeed={setDuelSpeedMs}
              onSetMaxTurns={setDuelMaxTurns}
            />
          )}
          <ComponentControls
            state={state}
            locked={aiTurn || currentPolicy !== "human"}
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
            onSetComponentCount={setComponentCount}
            onResetDefaults={() => dispatch({ type: "reset-default-components" })}
            onRestart={() => dispatch({ type: "restart", keepDefinitions: true })}
          />
          <WaveEditor
            definitions={state.definitions}
            componentCounts={{
              pawn: state.components[state.currentPlayer].pawn.length,
              rook: state.components[state.currentPlayer].rook.length,
              spy: state.components[state.currentPlayer].spy.length,
              king: state.components[state.currentPlayer].king.length,
            }}
            selected={editorSelection}
            onSelect={setEditorSelection}
            onSetComponentCount={setComponentCount}
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
