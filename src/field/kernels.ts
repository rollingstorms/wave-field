import type { BasisDefinition, Coefficient, FormulaPreset, Position } from "../game/types";

function ring(offset: Position): number {
  return Math.max(Math.abs(offset.x), Math.abs(offset.y));
}

function decay(r: number, decayBase: number, originScale: number): number {
  return Math.pow(decayBase, -r) * (r === 0 ? originScale : 1);
}

function presetSign(preset: FormulaPreset, delta: Position, r: number): Coefficient {
  switch (preset) {
    case "checkerboard":
      return (Math.abs(delta.x) + Math.abs(delta.y)) % 2 === 0 ? 1 : -1;
    case "diagonal-stripes":
      return Math.abs(delta.x - delta.y) % 2 === 0 ? 1 : -1;
    case "horizontal-versus-vertical":
      return Math.abs(delta.x) >= Math.abs(delta.y) ? 1 : -1;
    case "quadrants":
      return delta.x * delta.y >= 0 ? 1 : -1;
    case "constant-basin":
      return 1;
    case "skipped-rings":
      if (r % 6 === 0) return 1;
      if (r % 6 === 3) return -1;
      return 0;
  }
}

export function evaluateBasis(definition: BasisDefinition, delta: Position): number {
  const r = ring(delta);
  const multiplier = decay(r, definition.decayBase, definition.originScale);
  if (definition.kind === "preset") {
    return presetSign(definition.preset, delta, r) * multiplier;
  }

  const index = definition.repeat ? r % definition.ringValues.length : r;
  const sign = definition.ringValues[index] ?? 0;
  return sign * multiplier;
}
