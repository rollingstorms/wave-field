import { BOARD_CENTER, BOARD_SIZE, DEBUG_COMPONENT_COUNT_LIMITS } from "../game/constants";
import { evaluateComponentBasis } from "../field/kernels";
import { PIECE_TYPES, pieceName } from "../game/pieceLabels";
import type { BasisDefinition, Coefficient, ComponentDefinitions, FormulaPreset, PieceType } from "../game/types";

const pieceTypes: PieceType[] = PIECE_TYPES;
const presets: FormulaPreset[] = [
  "checkerboard",
  "diagonal-stripes",
  "horizontal-versus-vertical",
  "quadrants",
  "constant-basin",
  "skipped-rings",
  "compass-rose",
  "axis-favor",
  "diagonal-favor",
  "wide-bullseye",
  "pulse-gap",
  "block-checker",
  "diamond-core",
  "astigmatism",
  "local-flip",
  "adjacent-opinion",
  "sink",
  "deep-sink",
  "far-crown",
  "slow-governance",
  "dipole-x",
  "dipole-y",
];
const coefficients: Coefficient[] = [1, 0, -1];
const gridLimit = 8;

interface WaveEditorProps {
  definitions: ComponentDefinitions;
  componentCounts: Record<PieceType, number>;
  selected: { pieceType: PieceType; componentIndex: number };
  onSelect: (selection: { pieceType: PieceType; componentIndex: number }) => void;
  onSetComponentCount: (pieceType: PieceType, count: number) => void;
  onUpdate: (definition: BasisDefinition) => void;
  onResetSelected: () => void;
  onResetAll: () => void;
  onImport: (payload: unknown) => void;
}

function label(value: number) {
  return value > 0 ? "+" : value < 0 ? "-" : "0";
}

function valueLabel(value: number) {
  if (Math.abs(value) < 0.001) return "0";
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}

function kernelValues(pieceType: PieceType, definition: BasisDefinition) {
  return Array.from({ length: BOARD_SIZE }, (_, y) =>
    Array.from({ length: BOARD_SIZE }, (_, x) => evaluateComponentBasis(pieceType, definition, { x: x - BOARD_CENTER, y: y - BOARD_CENTER })),
  );
}

function editableRingValues(definition: BasisDefinition) {
  return definition.kind === "ring" ? definition.ringValues.slice(1) : [];
}

function blankGrid() {
  return Array.from({ length: BOARD_SIZE }, () => Array.from({ length: BOARD_SIZE }, () => 0));
}

function gridFromDefinition(pieceType: PieceType, definition: BasisDefinition) {
  if (definition.kind === "grid") return structuredClone(definition.gridValues);
  return Array.from({ length: BOARD_SIZE }, (_, y) =>
    Array.from({ length: BOARD_SIZE }, (_, x) => {
      if (x === BOARD_CENTER && y === BOARD_CENTER) return 0;
      return Math.max(-1, Math.min(1, Math.sign(evaluateComponentBasis(pieceType, definition, { x: x - BOARD_CENTER, y: y - BOARD_CENTER }))));
    }),
  );
}

function clampGridValue(value: number) {
  return Math.max(-gridLimit, Math.min(gridLimit, Math.trunc(value || 0)));
}

export function WaveEditor({ definitions, componentCounts, selected, onSelect, onSetComponentCount, onUpdate, onResetSelected, onResetAll, onImport }: WaveEditorProps) {
  const definition = definitions[selected.pieceType][selected.componentIndex];
  const grid = kernelValues(selected.pieceType, definition);
  const selectedCount = componentCounts[selected.pieceType];

  function updateRingValue(index: number, value: Coefficient) {
    const base: BasisDefinition = definition.kind === "ring"
      ? structuredClone(definition)
      : { kind: "ring", name: definition.name, geometry: "chebyshev", ringValues: [1, 1, -1, -1], repeat: true, decayBase: definition.decayBase, originScale: definition.originScale };
    if (base.kind === "ring") {
      base.ringValues[index] = value;
      onUpdate(base);
    }
  }

  function updateGridValue(x: number, y: number, value: number) {
    const gridValues = definition.kind === "grid" ? structuredClone(definition.gridValues) : gridFromDefinition(selected.pieceType, definition);
    gridValues[y][x] = x === BOARD_CENTER && y === BOARD_CENTER ? 0 : clampGridValue(value);
    onUpdate({
      kind: "grid",
      name: definition.name,
      gridValues,
      decayBase: definition.decayBase,
      originScale: definition.originScale,
    });
  }

  function addComponent(pieceType: PieceType) {
    const count = componentCounts[pieceType];
    if (count >= DEBUG_COMPONENT_COUNT_LIMITS[pieceType]) return;
    onSetComponentCount(pieceType, count + 1);
    onSelect({ pieceType, componentIndex: count });
  }

  function removeComponent(pieceType: PieceType) {
    const count = componentCounts[pieceType];
    if (count <= 1) return;
    const nextCount = count - 1;
    onSetComponentCount(pieceType, nextCount);
    onSelect({ pieceType, componentIndex: Math.min(selected.componentIndex, nextCount - 1) });
  }

  return (
    <section className="wave-editor">
      <div className="editor-bar">
        <button onClick={onResetSelected}>Reset selected</button>
        <button onClick={onResetAll}>Reset all</button>
        <button onClick={() => navigator.clipboard?.writeText(JSON.stringify(definitions, null, 2))}>Export JSON</button>
        <label className="import-button">
          Import JSON
          <input
            type="file"
            accept="application/json"
            onChange={async (event) => {
              const file = event.currentTarget.files?.[0];
              if (file) onImport(JSON.parse(await file.text()));
              event.currentTarget.value = "";
            }}
          />
        </label>
      </div>

      <div className="pattern-tabs" aria-label="Piece pattern tabs">
        {pieceTypes.map((pieceType) => (
          <button
            key={pieceType}
            className={pieceType === selected.pieceType ? "active" : ""}
            onClick={() => onSelect({ pieceType, componentIndex: Math.min(selected.componentIndex, componentCounts[pieceType] - 1) })}
          >
            {pieceName(pieceType).toUpperCase()}
          </button>
        ))}
      </div>

      <div className="component-tabs" aria-label={`${pieceName(selected.pieceType)} component tabs`}>
        {Array.from({ length: selectedCount }, (_, index) => (
          <button
            key={index}
            className={index === selected.componentIndex ? "active" : ""}
            onClick={() => onSelect({ ...selected, componentIndex: index })}
          >
            C{index + 1}
          </button>
        ))}
        <button
          aria-label={`Add ${pieceName(selected.pieceType)} component`}
          disabled={selectedCount >= DEBUG_COMPONENT_COUNT_LIMITS[selected.pieceType]}
          onClick={() => addComponent(selected.pieceType)}
        >
          +
        </button>
        <button
          aria-label={`Remove ${pieceName(selected.pieceType)} component`}
          disabled={selectedCount <= 1}
          onClick={() => removeComponent(selected.pieceType)}
        >
          -
        </button>
      </div>

      <div className="editor-grid">
        <div>
          <h3>Mode</h3>
          <div className="segmented">
            <button className={definition.kind === "preset" ? "active" : ""} onClick={() => onUpdate({ kind: "preset", name: definition.name, preset: "checkerboard", decayBase: definition.decayBase, originScale: definition.originScale })}>Preset</button>
            <button className={definition.kind === "ring" ? "active" : ""} onClick={() => onUpdate({ kind: "ring", name: definition.name, geometry: "chebyshev", ringValues: [1, 1, -1, -1], repeat: true, decayBase: definition.decayBase, originScale: definition.originScale })}>Rings</button>
            <button className={definition.kind === "grid" ? "active" : ""} onClick={() => onUpdate({ kind: "grid", name: definition.name, gridValues: gridFromDefinition(selected.pieceType, definition), decayBase: definition.decayBase, originScale: definition.originScale })}>Grid</button>
          </div>
          {definition.kind === "preset" ? (
            <select value={definition.preset} onChange={(event) => onUpdate({ ...definition, preset: event.target.value as FormulaPreset })}>
              {presets.map((preset) => <option key={preset} value={preset}>{preset}</option>)}
            </select>
          ) : definition.kind === "ring" ? (
            <div className="ring-editor">
              {editableRingValues(definition).map((value, index) => (
                <div className="ring-row" key={index + 1}>
                  <span>R{index + 1}</span>
                  {coefficients.map((coefficient) => (
                    <button key={coefficient} className={coefficient === value ? "active" : ""} onClick={() => updateRingValue(index + 1, coefficient)}>{label(coefficient)}</button>
                  ))}
                </div>
              ))}
            </div>
          ) : definition.kind === "grid" ? (
            <div className="grid-editor" aria-label="Raw integer pattern values">
              {(definition.gridValues.length === BOARD_SIZE ? definition.gridValues : blankGrid()).map((row, y) => row.map((value, x) => (
                <div
                  key={`${x}-${y}`}
                  className={`grid-stepper ${value > 0 ? "red" : value < 0 ? "blue" : "neutral"}`}
                  aria-label={`Pattern value ${x - BOARD_CENTER},${y - BOARD_CENTER}: ${x === BOARD_CENTER && y === BOARD_CENTER ? 0 : value}`}
                >
                  <button
                    aria-label={`Increase pattern value ${x - BOARD_CENTER},${y - BOARD_CENTER}`}
                    disabled={x === BOARD_CENTER && y === BOARD_CENTER || value >= gridLimit}
                    onClick={() => updateGridValue(x, y, value + 1)}
                  >
                    ▲
                  </button>
                  <span>{x === BOARD_CENTER && y === BOARD_CENTER ? 0 : value}</span>
                  <button
                    aria-label={`Decrease pattern value ${x - BOARD_CENTER},${y - BOARD_CENTER}`}
                    disabled={x === BOARD_CENTER && y === BOARD_CENTER || value <= -gridLimit}
                    onClick={() => updateGridValue(x, y, value - 1)}
                  >
                    ▼
                  </button>
                </div>
              )))}
            </div>
          ) : (
            <p>Combo patterns can be edited by importing JSON.</p>
          )}
        </div>
        <div>
          <h3>Kernel preview</h3>
          <div className="kernel">
            {grid.map((row, y) => row.map((value, x) => (
              <span className={value > 0 ? "red" : value < 0 ? "blue" : "neutral"} key={`${x}-${y}`}>{definition.kind === "grid" ? valueLabel(value) : label(value)}</span>
            )))}
          </div>
        </div>
      </div>
    </section>
  );
}
