import { BOARD_SIZE } from "../game/constants";
import { evaluateBasis } from "../field/kernels";
import type { BasisDefinition, Coefficient, ComponentDefinitions, FormulaPreset, PieceType } from "../game/types";

const pieceTypes: PieceType[] = ["pawn", "rook", "spy", "king"];
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
];
const coefficients: Coefficient[] = [1, 0, -1];

interface WaveEditorProps {
  definitions: ComponentDefinitions;
  selected: { pieceType: PieceType; componentIndex: number };
  onSelect: (selection: { pieceType: PieceType; componentIndex: number }) => void;
  onUpdate: (definition: BasisDefinition) => void;
  onResetSelected: () => void;
  onResetAll: () => void;
  onImport: (payload: unknown) => void;
}

function componentCount(pieceType: PieceType) {
  return definitionsShape[pieceType];
}

const definitionsShape = { pawn: 1, rook: 2, spy: 3, king: 3 } as const;

function label(value: number) {
  return value > 0 ? "+" : value < 0 ? "-" : "0";
}

function kernelValues(definition: BasisDefinition) {
  return Array.from({ length: BOARD_SIZE }, (_, y) =>
    Array.from({ length: BOARD_SIZE }, (_, x) => evaluateBasis(definition, { x: x - 3, y: y - 3 })),
  );
}

export function WaveEditor({ definitions, selected, onSelect, onUpdate, onResetSelected, onResetAll, onImport }: WaveEditorProps) {
  const definition = definitions[selected.pieceType][selected.componentIndex];
  const grid = kernelValues(definition);

  function updateRingValue(index: number, value: Coefficient) {
    const base: BasisDefinition = definition.kind === "ring"
      ? structuredClone(definition)
      : { kind: "ring", name: definition.name, geometry: "chebyshev", ringValues: [1, 1, -1, -1], repeat: true, decayBase: definition.decayBase, originScale: definition.originScale };
    if (base.kind === "ring") {
      base.ringValues[index] = value;
      onUpdate(base);
    }
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
            Array.from({ length: componentCount(pieceType) }, (_, index) => (
              <option key={`${pieceType}-${index}`} value={`${pieceType}:${index}`}>{pieceType.toUpperCase()} C{index + 1}</option>
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
          </div>
          {definition.kind === "preset" ? (
            <select value={definition.preset} onChange={(event) => onUpdate({ ...definition, preset: event.target.value as FormulaPreset })}>
              {presets.map((preset) => <option key={preset} value={preset}>{preset}</option>)}
            </select>
          ) : (
            <div className="ring-editor">
              {definition.ringValues.map((value, index) => (
                <div className="ring-row" key={index}>
                  <span>R{index}</span>
                  {coefficients.map((coefficient) => (
                    <button key={coefficient} className={coefficient === value ? "active" : ""} onClick={() => updateRingValue(index, coefficient)}>{label(coefficient)}</button>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>
        <div>
          <h3>Kernel preview</h3>
          <div className="kernel">
            {grid.map((row, y) => row.map((value, x) => (
              <span className={value > 0 ? "red" : value < 0 ? "blue" : "neutral"} key={`${x}-${y}`}>{label(value)}</span>
            )))}
          </div>
        </div>
      </div>
    </section>
  );
}
