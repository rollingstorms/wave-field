import type { BasisDefinition, Coefficient, FormulaPreset, PieceType, Position } from "../game/types";

function ring(offset: Position): number {
  return Math.max(Math.abs(offset.x), Math.abs(offset.y));
}

function decay(r: number, decayBase: number, originScale: number): number {
  return Math.pow(decayBase, -r) * (r === 0 ? originScale : 1);
}

function presetSign(preset: FormulaPreset, delta: Position, r: number): Coefficient {
  const absX = Math.abs(delta.x);
  const absY = Math.abs(delta.y);

  switch (preset) {
    case "checkerboard":
      return (absX + absY) % 2 === 0 ? 1 : -1;
    case "diagonal-stripes":
      return Math.floor(Math.abs(delta.x - delta.y) / 2) % 2 === 0 ? 1 : -1;
    case "horizontal-versus-vertical":
      return absX >= absY ? 1 : -1;
    case "quadrants":
      return delta.x * delta.y >= 0 ? 1 : -1;
    case "constant-basin":
      return 1;
    case "skipped-rings":
      if (r % 6 === 0) return 1;
      if (r % 6 === 3) return -1;
      return 0;
    case "compass-rose":
      return delta.x === 0 || delta.y === 0 || absX === absY ? 1 : -1;
    case "axis-favor":
      if (delta.x === 0 || delta.y === 0) return 1;
      if (absX === absY) return -1;
      return 0;
    case "diagonal-favor":
      if (absX === absY) return 1;
      if (delta.x === 0 || delta.y === 0) return -1;
      return 0;
    case "wide-bullseye":
      return Math.floor(r / 2) % 2 === 0 ? 1 : -1;
    case "pulse-gap":
      if (r % 4 === 0) return 1;
      if (r % 4 === 2) return -1;
      return 0;
    case "block-checker":
      return (Math.floor(absX / 2) + Math.floor(absY / 2)) % 2 === 0 ? 1 : -1;
    case "diamond-core":
      return absX + absY <= 2 ? 1 : -1;
    case "astigmatism":
      if (absX === absY) return 0;
      return absX > absY ? 1 : -1;
    case "local-flip":
      return r <= 1 ? 1 : 0;
    case "adjacent-opinion":
      return r === 1 ? 1 : 0;
    case "sink":
      return r === 0 ? 1 : -1;
    case "deep-sink":
      return r <= 1 ? 1 : -1;
    case "far-crown":
      return r >= 3 ? 1 : 0;
    case "slow-governance":
      return r <= 2 ? 1 : -1;
    case "dipole-x":
      if (delta.x === 0) return 0;
      return delta.x > 0 ? 1 : -1;
    case "dipole-y":
      if (delta.y === 0) return 0;
      return delta.y > 0 ? 1 : -1;
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

export function evaluateComponentBasis(pieceType: PieceType, definition: BasisDefinition, delta: Position): number {
  if (pieceType === "king" && delta.x === 0 && delta.y === 0) return 0;
  return evaluateBasis(definition, delta);
}
