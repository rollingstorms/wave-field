import { describe, expect, it } from "vitest";
import { DEFAULT_WAVE_SCALES } from "../game/constants";
import {
  candidateDefinitionVariants,
  componentPatternMetrics,
  enumerateDefaultComponentSets,
  enumerateProfiles,
  evaluateCombinedParameterCandidate,
  evaluateDefaultComponentSet,
  evaluateDefinitionSetVariant,
  evaluateHomeEnergySet,
  evaluateWaveScaleSet,
  findComboOutliers,
  profileMobilityDiagnostics,
  profilePowerMetrics,
  searchDefaultComponentSets,
  searchCombinedParameterCandidates,
  searchDefinitionVariants,
  searchHomeEnergyOptions,
  searchWaveScaleOptions,
} from "../game/mechanicsAnalysis";

describe("mechanics analysis", () => {
  it("enumerates legal full-strength profiles", () => {
    expect(enumerateProfiles("pawn")).toHaveLength(2);
    expect(enumerateProfiles("rook")).toHaveLength(4);
    expect(enumerateProfiles("spy")).toHaveLength(6);
    expect(enumerateProfiles("king")).toHaveLength(4);
  });

  it("measures polarity imbalance created by scales and clipped board geometry", () => {
    const metrics = profilePowerMetrics();

    expect(metrics).toHaveLength(16);
    expect(metrics.some((metric) => Math.abs(metric.polarityL1Ratio - 1) > 0.01)).toBe(true);
    expect(metrics.some((metric) => Math.abs(metric.polarityNetDelta) > 0.01)).toBe(true);
  });

  it("diagnoses raw component patterns and first-step mobility", () => {
    const patterns = componentPatternMetrics();
    const mobility = profileMobilityDiagnostics();
    const deadRook = mobility.find((row) =>
      row.pieceType === "rook" && row.profile.every((value) => value === -1));

    expect(patterns).toHaveLength(9);
    expect(patterns.some((metric) => metric.adjacentPositive + metric.adjacentNegative + metric.adjacentZero > 0)).toBe(true);
    expect(mobility).toHaveLength(16);
    expect(deadRook?.averageMoves).toBeGreaterThan(10);
    expect(deadRook?.deadOrigins).toBe(0);
    expect(mobility.find((row) =>
      row.pieceType === "rook" && row.profile.join(",") === "1,-1")?.deadOrigins).toBe(49);
  });

  it("finds pairwise field outliers", () => {
    const outliers = findComboOutliers(undefined, 3, {
      positions: [
        { x: 1, y: 1 },
        { x: 3, y: 1 },
        { x: 3, y: 3 },
        { x: 1, y: 5 },
        { x: 5, y: 5 },
      ],
    });

    expect(outliers.reinforcement).toHaveLength(3);
    expect(outliers.cancellation).toHaveLength(3);
    expect(outliers.fragmentation).toHaveLength(3);
    expect(outliers.bias).toHaveLength(3);
    expect(outliers.fragmentation[0].metrics.edgeCount).toBeGreaterThan(0);
  });

  it("scores and ranks wave scale candidates", () => {
    const current = evaluateWaveScaleSet(DEFAULT_WAVE_SCALES);
    const candidates = searchWaveScaleOptions({
      pawn: [{ friendly: 4, hostile: 1 }, { friendly: 3, hostile: 2 }],
      rook: [{ friendly: 3, hostile: 1 }, { friendly: 2, hostile: 1.5 }],
      spy: [{ friendly: 3, hostile: 0 }, { friendly: 2, hostile: 1 }],
      king: [{ friendly: 4, hostile: 2 }, { friendly: 3, hostile: 2 }],
    }, 3);

    expect(current.maxPolarityRatio).toBeGreaterThan(1);
    expect(candidates).toHaveLength(3);
    expect(candidates[0].score).toBeLessThanOrEqual(candidates[1].score);
    expect(candidates[0].deadProfiles.length).toBeLessThanOrEqual(current.deadProfiles.length);
  });

  it("scores and ranks default component sets", () => {
    const allDefaults = enumerateDefaultComponentSets();
    const current = evaluateDefaultComponentSet({
      pawn: [1],
      rook: [1, 1],
      spy: [1, 0, 0],
      king: [1, 1],
    });
    const candidates = searchDefaultComponentSets(3, undefined, [
      current.components,
      { pawn: [1], rook: [1, -1], spy: [0, -1, 0], king: [-1, 1] },
      { pawn: [-1], rook: [-1, -1], spy: [-1, 0, 0], king: [-1, -1] },
    ]);

    expect(allDefaults).toHaveLength(192);
    expect(current.openingMoves.blue).toBe(current.openingMoves.red);
    expect(candidates).toHaveLength(3);
    expect(candidates[0].score).toBeLessThanOrEqual(candidates[1].score);
  });

  it("scores and ranks home energy candidates", () => {
    const current = evaluateHomeEnergySet({ pawn: 0, rook: 0, spy: 0.5, king: 0 });
    const candidates = searchHomeEnergyOptions({
      pawn: [0, 0.25],
      rook: [0, 0.25],
      spy: [0, 0.5],
      king: [0, 0.25],
    }, 3);

    expect(current.bigHatMargins.blue).toBeCloseTo(current.bigHatMargins.red);
    expect(candidates).toHaveLength(3);
    expect(candidates[0].score).toBeLessThanOrEqual(candidates[1].score);
  });

  it("scores and ranks pattern definition variants", () => {
    const variants = searchDefinitionVariants([
      { pieceType: "rook", componentIndex: 0 },
      { pieceType: "king", componentIndex: 2 },
    ], 4);

    expect(variants).toHaveLength(4);
    expect(variants[0].score).toBeLessThanOrEqual(variants[1].score);
    expect(variants.some((variant) => variant.adjacentCounts.negative > 0 || variant.adjacentCounts.zero > 0)).toBe(true);
  });

  it("scores coupled pattern definition variants and reports dead profiles", () => {
    const rookC1 = candidateDefinitionVariants("rook", 0)[0];
    const rookC2 = candidateDefinitionVariants("rook", 1).find((definition) =>
      definition.kind === "ring" && definition.ringValues.join(",") === "0,1,0,-1");
    expect(rookC1).toBeDefined();
    expect(rookC2).toBeDefined();

    const variant = evaluateDefinitionSetVariant([
      { pieceType: "rook", componentIndex: 0, definition: rookC1! },
      { pieceType: "rook", componentIndex: 1, definition: rookC2! },
    ]);

    expect(variant.replacements).toHaveLength(2);
    expect(variant.deadProfiles).toHaveLength(1);
    expect(variant.minAverageMoves).toBe(0);
  });

  it("scores combined parameter candidates", () => {
    const current = evaluateCombinedParameterCandidate({ name: "current" });
    const rookC1 = candidateDefinitionVariants("rook", 0)[0];
    const rookC2 = candidateDefinitionVariants("rook", 1).find((definition) =>
      definition.kind === "ring" && definition.ringValues.join(",") === "0,1,0,-1");
    expect(rookC1).toBeDefined();
    expect(rookC2).toBeDefined();

    const candidates = searchCombinedParameterCandidates([
      { name: "current" },
      {
        name: "rook-neutral-first-ring",
        replacements: [
          { pieceType: "rook", componentIndex: 0, definition: rookC1! },
          { pieceType: "rook", componentIndex: 1, definition: rookC2! },
        ],
      },
    ], 2);

    expect(current.wave.deadProfiles.length).toBeGreaterThan(0);
    expect(candidates).toHaveLength(2);
    expect(candidates[0].score).toBeLessThanOrEqual(candidates[1].score);
  });
});
