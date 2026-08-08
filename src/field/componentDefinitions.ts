import type { ComponentDefinitions, PlayerComponents } from "../game/types";
import { BOARD_SIZE, DEBUG_COMPONENT_COUNT_LIMITS, DEFAULT_COMPONENT_COUNTS, WAVE_DECAY_BASE, WAVE_ORIGIN_SCALE } from "../game/constants";
import type { BasisDefinition, FormulaPreset } from "../game/types";

export const DEFAULT_COMPONENTS: PlayerComponents = {
  pawn: [1],
  rook: [1, 1],
  spy: [1, 0],
  king: [1, 1],
};

export const TRAINING_COMPONENTS: PlayerComponents = {
  pawn: [1],
  rook: [1, 1],
  spy: [1, 0, 0],
  king: [0, 1, 1],
};

export const DEBUG_DEFINITIONS: ComponentDefinitions = {
  pawn: [{ kind: "preset", name: "Checkerboard", preset: "checkerboard", decayBase: 2, originScale: 1 }],
  rook: [
    { kind: "ring", name: "Push pairs", geometry: "chebyshev", ringValues: [0, 0, 1, -1], repeat: true, decayBase: 2, originScale: 1 },
    { kind: "ring", name: "Pull gap push", geometry: "chebyshev", ringValues: [0, 1, 0, -1], repeat: true, decayBase: 2, originScale: 1 },
  ],
  spy: [
    { kind: "preset", name: "Round Hat mask", preset: "checkerboard", decayBase: 2, originScale: 1 },
    { kind: "preset", name: "Diamond core", preset: "diamond-core", decayBase: 2, originScale: 1 },
    { kind: "preset", name: "Axis favor", preset: "axis-favor", decayBase: 2, originScale: 1 },
  ],
  king: [
    { kind: "ring", name: "Slow alternating rings", geometry: "chebyshev", ringValues: [1, 1, 1, -1, -1, -1], repeat: true, decayBase: 2, originScale: 1 },
    { kind: "preset", name: "Horizontal mode", preset: "horizontal-versus-vertical", decayBase: 2, originScale: 1 },
    { kind: "preset", name: "Block checker", preset: "block-checker", decayBase: 2, originScale: 1 },
  ],
};

export function definitionsForCounts(counts = DEFAULT_COMPONENT_COUNTS): ComponentDefinitions {
  return {
    pawn: DEBUG_DEFINITIONS.pawn.slice(0, counts.pawn),
    rook: DEBUG_DEFINITIONS.rook.slice(0, counts.rook),
    spy: DEBUG_DEFINITIONS.spy.slice(0, counts.spy),
    king: DEBUG_DEFINITIONS.king.slice(0, counts.king),
  };
}

export const DEFAULT_DEFINITIONS: ComponentDefinitions = DEBUG_DEFINITIONS;

export function definitionForSlot(pieceType: keyof ComponentDefinitions, componentIndex: number): BasisDefinition {
  return structuredClone(DEBUG_DEFINITIONS[pieceType][componentIndex] ?? DEBUG_DEFINITIONS[pieceType][0]);
}

export function cloneDefinitions(definitions = DEFAULT_DEFINITIONS): ComponentDefinitions {
  return JSON.parse(JSON.stringify(definitions)) as ComponentDefinitions;
}

const formulaPresets: FormulaPreset[] = [
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

export function validateDefinition(value: unknown): value is BasisDefinition {
  if (!value || typeof value !== "object") return false;
  const definition = value as BasisDefinition;
  if (
    typeof definition.name !== "string"
    || definition.decayBase !== WAVE_DECAY_BASE
    || definition.originScale !== WAVE_ORIGIN_SCALE
  ) {
    return false;
  }
  if (definition.kind === "preset") {
    return formulaPresets.includes(definition.preset);
  }
  if (definition.kind === "ring") {
    return definition.geometry === "chebyshev"
      && typeof definition.repeat === "boolean"
      && Array.isArray(definition.ringValues)
      && definition.ringValues.length > 0
      && definition.ringValues.every((coefficient) => coefficient === -1 || coefficient === 0 || coefficient === 1);
  }
  if (definition.kind === "grid") {
    return Array.isArray(definition.gridValues)
      && definition.gridValues.length === BOARD_SIZE
      && definition.gridValues.every((row) =>
        Array.isArray(row)
        && row.length === BOARD_SIZE
        && row.every((value) => Number.isInteger(value) && Math.abs(value) <= 8));
  }
  if (definition.kind === "combo") {
    return Array.isArray(definition.components)
      && definition.components.length > 0
      && definition.components.every((component) =>
        Number.isFinite(component.weight)
        && Math.abs(component.weight) <= 4
        && validateDefinition(component.definition));
  }
  return false;
}

export function validateDefinitions(value: unknown, counts?: Record<keyof ComponentDefinitions, number>): value is ComponentDefinitions {
  if (!value || typeof value !== "object") return false;
  const record = value as ComponentDefinitions;
  const requiredCounts = counts;
  return (Object.keys(DEBUG_COMPONENT_COUNT_LIMITS) as Array<keyof typeof DEBUG_COMPONENT_COUNT_LIMITS>).every((type) => {
    const defs = record[type];
    const validLength = requiredCounts
      ? defs?.length === requiredCounts[type]
      : Array.isArray(defs) && defs.length >= 1 && defs.length <= DEBUG_COMPONENT_COUNT_LIMITS[type];
    return Array.isArray(defs) && validLength && defs.every(validateDefinition);
  });
}
