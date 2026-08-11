import { cloneDefinitions, DEFAULT_COMPONENTS } from "../field/componentDefinitions";
import { evaluateField } from "../field/evaluateField";
import { projectFieldValue } from "../field/projection";
import { DEFAULT_HOME_ENERGY, DEFAULT_WAVE_SCALES } from "./constants";
import { createInitialState } from "./initialState";
import type { ComponentDefinitions, HomeEnergy, PlayerComponents, WaveScales } from "./types";

export interface OptimizedGameConfig {
  name: string;
  components: PlayerComponents;
  definitions: ComponentDefinitions;
  waveScales: WaveScales;
  homeEnergy: HomeEnergy;
}

export type GameVariantId = "balanced-pressure" | "easy-mobility";

export function openingTerritoryRowsMatchOriginal(config: OptimizedGameConfig): boolean {
  const state = createInitialState(config.components, config.definitions, config.waveScales, config.homeEnergy);
  const field = evaluateField(state);
  return field.every((row, y) =>
    row.every((value) => {
      const territory = projectFieldValue(value);
      if (y < 3) return territory === "blue";
      if (y === 3) return territory === "neutral";
      return territory === "red";
    }),
  );
}

function createBalancedPressureConfig(): OptimizedGameConfig {
  const definitions = cloneDefinitions();
  definitions.rook[0] = {
    ...definitions.rook[0],
    name: "Optim Tower push pairs",
    kind: "ring",
    geometry: "chebyshev",
    ringValues: [0, 0, 1, -1],
    repeat: true,
  };
  definitions.rook[1] = {
    ...definitions.rook[1],
    name: "Optim Tower gap push",
    kind: "ring",
    geometry: "chebyshev",
    ringValues: [0, 1, 0, -1],
    repeat: true,
  };
  definitions.spy[1] = {
    ...definitions.spy[1],
    name: "Optim Triangle diagonal favor",
    kind: "preset",
    preset: "diagonal-favor",
  };
  definitions.king[0] = {
    ...definitions.king[0],
    name: "Optim Big Hat alternating rings",
    kind: "ring",
    geometry: "chebyshev",
    ringValues: [1, 1, -1, 1],
    repeat: true,
  };
  definitions.king[2] = {
    ...definitions.king[2],
    name: "Optim Big Hat block checker",
    kind: "preset",
    preset: "block-checker",
  };

  return {
    name: "optim-test-balanced-pressure",
    components: structuredClone(DEFAULT_COMPONENTS),
    definitions,
    waveScales: {
      ...structuredClone(DEFAULT_WAVE_SCALES),
      rook: { friendly: 2, hostile: 0 },
      king: { friendly: 3, hostile: 0 },
    },
    homeEnergy: structuredClone(DEFAULT_HOME_ENERGY),
  };
}

function createEasyMobilityConfig(): OptimizedGameConfig {
  const definitions = cloneDefinitions();
  definitions.spy[1] = {
    ...definitions.spy[1],
    name: "Easy Triangle diagonal favor",
    kind: "preset",
    preset: "diagonal-favor",
  };
  definitions.king[2] = {
    ...definitions.king[2],
    name: "Easy Big Hat astigmatism",
    kind: "preset",
    preset: "astigmatism",
  };

  return {
    name: "easy-mobility",
    components: structuredClone(DEFAULT_COMPONENTS),
    definitions,
    waveScales: structuredClone(DEFAULT_WAVE_SCALES),
    homeEnergy: structuredClone(DEFAULT_HOME_ENERGY),
  };
}

export function createGameVariantConfig(variant: GameVariantId): OptimizedGameConfig {
  switch (variant) {
    case "easy-mobility":
      return createEasyMobilityConfig();
    case "balanced-pressure":
      return createBalancedPressureConfig();
  }
}

export function createOptimTestConfig(): OptimizedGameConfig {
  return createGameVariantConfig("balanced-pressure");
}
