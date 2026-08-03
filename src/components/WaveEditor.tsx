import { BOARD_SIZE } from "../game/constants";
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
    Array.from({ length: BOARD_SIZE }, (_, x) => evaluateComponentBasis(pieceType, definition, { x: x - 3, y: y - 3 })),
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
      if (x === 3 && y === 3) return 0;
      return Math.max(-1, Math.min(1, Math.sign(evaluateComponentBasis(pieceType, definition, { x: x - 3, y: y - 3 }))));
    }),
  );
}

function clampGridValue(value: number) {
  return Math.max(-gridLimit, Math.min(gridLimit, Math.trunc(value || 0)));
}

export function WaveEditor({ definitions, componentCounts, selected, onSelect, onUpdate, onResetSelected, onResetAll, onImport }: WaveEditorProps) {
  const definition = definitions[selected.pieceType][selected.componentIndex];
  const grid = kernelValues(selected.pieceType, definition);

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
    gridValues[y][x] = x === 3 && y === 3 ? 0 : clampGridValue(value);
    onUpdate({
      kind: "grid",
      name: definition.name,
      gridValues,
      decayBase: definition.decayBase,
      originScale: definition.originScale,
    });
  }

  return (
    <section className="wave-editor">
      <div className="editor-bar">
        <select
          value={`${selected.pieceType}:${selected.componentIndex}`}
          onChange={(event) => {
            const [pieceType, componentIndex] = event.target.value.split(":");
            onSelect({ pieceType: pieceType as PieceType, componentIndex: Number(componentIndex) });
          }}
          aria-label="Component selector"
        >
          {pieceTypes.flatMap((pieceType) =>
            Array.from({ length: componentCounts[pieceType] }, (_, index) => (
              <option key={`${pieceType}-${index}`} value={`${pieceType}:${index}`}>{pieceName(pieceType).toUpperCase()} C{index + 1}</option>
            )),
          )}
        </select>
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
                <input
                  key={`${x}-${y}`}
                  type="number"
                  min={-gridLimit}
                  max={gridLimit}
                  step={1}
                  value={x === 3 && y === 3 ? 0 : value}
                  disabled={x === 3 && y === 3}
                  aria-label={`Pattern value ${x - 3},${y - 3}`}
                  className={value > 0 ? "red" : value < 0 ? "blue" : "neutral"}
                  onChange={(event) => updateGridValue(x, y, Number(event.target.value))}
                />
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
