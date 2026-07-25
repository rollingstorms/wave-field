import type { ComponentDefinitions, PlayerComponents } from "../game/types";

export const DEFAULT_COMPONENTS: PlayerComponents = {
  pawn: [1],
  rook: [1, 1],
  spy: [1, 0, 0],
  king: [1, 0, 0, 0],
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
    { kind: "preset", name: "Constant basin", preset: "constant-basin", decayBase: 2, originScale: 1 },
    { kind: "ring", name: "Slow alternating rings", geometry: "chebyshev", ringValues: [1, 1, 1, -1, -1, -1], repeat: true, decayBase: 2, originScale: 1 },
    { kind: "preset", name: "Horizontal mode", preset: "horizontal-versus-vertical", decayBase: 2, originScale: 1 },
    { kind: "preset", name: "Diagonal mode", preset: "quadrants", decayBase: 2, originScale: 1 },
  ],
};

export function cloneDefinitions(definitions = DEFAULT_DEFINITIONS): ComponentDefinitions {
  return JSON.parse(JSON.stringify(definitions)) as ComponentDefinitions;
}

export function validateDefinitions(value: unknown): value is ComponentDefinitions {
  if (!value || typeof value !== "object") return false;
  const record = value as ComponentDefinitions;
  const counts = { pawn: 1, rook: 2, spy: 3, king: 4 };
  return (Object.keys(counts) as Array<keyof typeof counts>).every((type) => {
    const defs = record[type];
    return Array.isArray(defs) && defs.length === counts[type] && defs.every((def) => {
      if (!def || typeof def !== "object" || typeof def.name !== "string") return false;
      if (typeof def.decayBase !== "number" || typeof def.originScale !== "number") return false;
      if (def.kind === "preset") return typeof def.preset === "string";
      if (def.kind === "ring") return Array.isArray(def.ringValues) && def.ringValues.every((v) => v === -1 || v === 0 || v === 1);
      return false;
    });
  });
}
