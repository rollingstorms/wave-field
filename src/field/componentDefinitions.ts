import type { ComponentDefinitions, PlayerComponents } from "../game/types";
import { WAVE_DECAY_BASE, WAVE_ORIGIN_SCALE } from "../game/constants";
import type { BasisDefinition, FormulaPreset } from "../game/types";

export const DEFAULT_COMPONENTS: PlayerComponents = {
  pawn: [1],
  rook: [1, 1],
  spy: [1, 0, 0],
  king: [1, 0, 0],
};

export const DEFAULT_DEFINITIONS: ComponentDefinitions = {
  pawn: [{ kind: "preset", name: "Checkerboard", preset: "checkerboard", decayBase: 2, originScale: 1 }],
  rook: [
    { kind: "ring", name: "Paired rings", geometry: "chebyshev", ringValues: [1, 1, -1, -1], repeat: true, decayBase: 2, originScale: 1 },
    { kind: "ring", name: "Shifted paired rings", geometry: "chebyshev", ringValues: [1, -1, -1, 1], repeat: true, decayBase: 2, originScale: 1 },
  ],
  spy: [
    { kind: "preset", name: "Pawn mask", preset: "checkerboard", decayBase: 2, originScale: 1 },
    { kind: "preset", name: "Diagonal stripes", preset: "diagonal-stripes", decayBase: 2, originScale: 1 },
    { kind: "preset", name: "Skipped rings", preset: "skipped-rings", decayBase: 2, originScale: 1 },
  ],
  king: [
    { kind: "ring", name: "Slow alternating rings", geometry: "chebyshev", ringValues: [1, 1, 1, -1, -1, -1], repeat: true, decayBase: 2, originScale: 1 },
    { kind: "preset", name: "Horizontal mode", preset: "horizontal-versus-vertical", decayBase: 2, originScale: 1 },
    { kind: "preset", name: "Diagonal mode", preset: "quadrants", decayBase: 2, originScale: 1 },
  ],
};

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
  return false;
}

export function validateDefinitions(value: unknown): value is ComponentDefinitions {
  if (!value || typeof value !== "object") return false;
  const record = value as ComponentDefinitions;
  const counts = { pawn: 1, rook: 2, spy: 3, king: 3 };
  return (Object.keys(counts) as Array<keyof typeof counts>).every((type) => {
    const defs = record[type];
    return Array.isArray(defs) && defs.length === counts[type] && defs.every(validateDefinition);
  });
}
