import { build } from "esbuild";
import { pathToFileURL } from "node:url";

const full = process.argv.includes("--full");
const outfile = "/tmp/wave-field-mechanics-report.mjs";

await build({
  stdin: {
    contents: `
      if (!globalThis.structuredClone) {
        globalThis.structuredClone = (value) => JSON.parse(JSON.stringify(value));
      }

      import {
        candidateDefinitionVariants,
        complexitySnapshot,
        componentPatternMetrics,
        enumerateProfiles,
        evaluateDefaultComponentSet,
        evaluateHomeEnergySet,
        evaluateWaveScaleSet,
        findComboOutliers,
        lossMobilityImpacts,
        legalMoveMobilityForProfile,
        mobilitySummary,
        moveConsequenceMetrics,
        profileMobilityDiagnostics,
        profilePowerMetrics,
        searchDefaultComponentSets,
        searchDefinitionSetVariants,
        searchDefinitionVariants,
        searchHomeEnergyOptions,
        searchCombinedParameterCandidates,
        searchWaveScaleOptions,
      } from "./src/game/mechanicsAnalysis.ts";
      import { DEFAULT_HOME_ENERGY, DEFAULT_WAVE_SCALES } from "./src/game/constants.ts";
      import { DEFAULT_COMPONENTS } from "./src/field/componentDefinitions.ts";

      const full = ${JSON.stringify(full)};
      const sampledPositions = [
        { x: 1, y: 1 },
        { x: 3, y: 1 },
        { x: 5, y: 1 },
        { x: 1, y: 3 },
        { x: 3, y: 3 },
        { x: 5, y: 3 },
        { x: 1, y: 5 },
        { x: 3, y: 5 },
        { x: 5, y: 5 },
      ];

      function profileLabel(profile) {
        return "[" + profile.map((value) => value > 0 ? "+" : value < 0 ? "-" : "0").join(" ") + "]";
      }

      function round(value) {
        return Math.round(value * 1000) / 1000;
      }

      function scalesLabel(scales) {
        return [
          "pawn=" + scales.pawn.friendly + "/" + scales.pawn.hostile,
          "rook=" + scales.rook.friendly + "/" + scales.rook.hostile,
          "spy=" + scales.spy.friendly + "/" + scales.spy.hostile,
          "king=" + scales.king.friendly + "/" + scales.king.hostile,
        ].join(" ");
      }

      function componentsLabel(components) {
        return [
          "pawn=" + profileLabel(components.pawn),
          "rook=" + profileLabel(components.rook),
          "spy=" + profileLabel(components.spy),
          "king=" + profileLabel(components.king),
        ].join(" ");
      }

      function homeEnergyLabel(homeEnergy) {
        return [
          "pawn=" + homeEnergy.pawn,
          "rook=" + homeEnergy.rook,
          "spy=" + homeEnergy.spy,
          "king=" + homeEnergy.king,
        ].join(" ");
      }

      function pieceLabel(pieceId) {
        return pieceId.replace("blue-", "B ").replace("red-", "R ").replace(/-/g, " ");
      }

      function destinationLabel(destination) {
        return String.fromCharCode(65 + destination.x) + (7 - destination.y);
      }

      function ringVariant(pieceType, componentIndex, values) {
        const wanted = values.join(",");
        const match = candidateDefinitionVariants(pieceType, componentIndex).find((definition) =>
          definition.kind === "ring" && definition.ringValues.join(",") === wanted
        );
        if (!match) throw new Error("Missing ring variant " + pieceType + " C" + componentIndex + " " + wanted);
        return match;
      }

      function presetVariant(pieceType, componentIndex, preset) {
        const match = candidateDefinitionVariants(pieceType, componentIndex).find((definition) =>
          definition.kind === "preset" && definition.preset === preset
        );
        if (!match) throw new Error("Missing preset variant " + pieceType + " C" + componentIndex + " " + preset);
        return match;
      }

      function resultLine(result) {
        return [
          "score=" + round(result.score),
          "maxRatio=" + round(result.maxPolarityRatio),
          "dead=" + result.deadProfiles.length,
          "minMob=" + round(result.minMobility),
          "spread=" + round(result.mobilitySpread),
          scalesLabel(result.scales),
          "worst=" + result.maxPolarityProfile.pieceType + profileLabel(result.maxPolarityProfile.profile) + ":" + round(result.maxPolarityProfile.ratio),
        ].join("  ");
      }

      function deadProfileLine(result) {
        if (result.deadProfiles.length === 0) return "deadProfiles=none";
        return "deadProfiles=" + result.deadProfiles.map((row) =>
          row.pieceType + profileLabel(row.profile) + ":" + round(row.mobility)
        ).join(", ");
      }

      const metrics = profilePowerMetrics();
      console.log("Profile polarity imbalance");
      for (const metric of [...metrics].sort((left, right) =>
        Math.abs(Math.log(right.polarityL1Ratio)) - Math.abs(Math.log(left.polarityL1Ratio))
      ).slice(0, 12)) {
        console.log([
          metric.pieceType.padEnd(5),
          profileLabel(metric.profile).padEnd(9),
          "l1Ratio=" + round(metric.polarityL1Ratio),
          "netDelta=" + round(metric.polarityNetDelta),
          "signBalance=" + round(metric.signBalance),
          "maxAbs=" + round(metric.maxAbs),
        ].join("  "));
      }

      console.log("\\nComponent pattern diagnostics");
      for (const metric of componentPatternMetrics().sort((left, right) =>
        Math.abs(right.netMass) - Math.abs(left.netMass)
      )) {
        console.log([
          metric.pieceType.padEnd(5),
          ("C" + (metric.componentIndex + 1)).padEnd(2),
          metric.name.padEnd(24),
          "net=" + round(metric.netMass),
          "balance=" + round(metric.signBalance),
          "cells=" + metric.positiveCells + "/" + metric.negativeCells + "/" + metric.zeroCells,
          "adj=" + metric.adjacentPositive + "/" + metric.adjacentNegative + "/" + metric.adjacentZero,
        ].join("  "));
      }

      console.log("\\nAverage solo mobility by profile");
      const mobilityDiagnostics = profileMobilityDiagnostics();
      for (const pieceType of ["pawn", "rook", "spy", "king"]) {
        const rows = mobilityDiagnostics
          .filter((row) => row.pieceType === pieceType)
          .sort((left, right) => right.averageMoves - left.averageMoves);
        for (const row of rows) {
          console.log([
            pieceType.padEnd(5),
            profileLabel(row.profile).padEnd(9),
            "moves/origin=" + round(row.averageMoves),
            "firstSteps=" + round(row.averageFirstSteps),
            "deadOrigins=" + row.deadOrigins,
            "firstStepRange=" + row.minFirstSteps + "-" + row.maxFirstSteps,
          ].join("  "));
        }
      }

      console.log("\\nCurrent-position complexity snapshot");
      const snapshot = complexitySnapshot(undefined, 6);
      const currentMobility = snapshot.mobility;
      console.log([
        "mobilityTotal=" + currentMobility.total,
        "byPlayer=B" + currentMobility.byPlayer.blue + "/R" + currentMobility.byPlayer.red,
        "byType=pawn:" + currentMobility.byType.pawn + " rook:" + currentMobility.byType.rook + " spy:" + currentMobility.byType.spy + " king:" + currentMobility.byType.king,
        "avgMargin=" + round(snapshot.averageSafetyMargin),
        "minMargin=" + round(snapshot.minSafetyMargin),
        "nearZeroPieces=" + snapshot.nearZeroPieceCount,
        "unstable=" + snapshot.unstablePieces,
      ].join("  "));
      console.log([
        "fragmentation",
        "cells=R" + snapshot.fragmentation.redCells + "/B" + snapshot.fragmentation.blueCells + "/N" + snapshot.fragmentation.neutralCells,
        "regions=R" + snapshot.fragmentation.redRegions + "/B" + snapshot.fragmentation.blueRegions + "/N" + snapshot.fragmentation.neutralRegions,
        "edges=" + snapshot.fragmentation.signEdges,
        "largest=" + snapshot.fragmentation.largestRegion,
      ].join("  "));
      console.log([
        "moveConsequences",
        "count=" + snapshot.moveConsequences.count,
        "avgSignChanges=" + round(snapshot.moveConsequences.averageSignChanges),
        "maxSignChanges=" + snapshot.moveConsequences.maxSignChanges,
        "avgL1=" + round(snapshot.moveConsequences.averageFieldL1Delta),
        "maxL1=" + round(snapshot.moveConsequences.maxFieldL1Delta),
        "avgMobSwing=" + round(snapshot.moveConsequences.averageMobilitySwing),
        "maxMobSwing=" + snapshot.moveConsequences.maxMobilitySwing,
      ].join("  "));

      console.log("\\nMobility per piece in the current opening");
      for (const row of [...currentMobility.pieces].sort((left, right) => right.legalMoves - left.legalMoves)) {
        console.log([
          pieceLabel(row.pieceId).padEnd(12),
          "moves=" + String(row.legalMoves).padStart(2),
          "margin=" + round(row.safetyMargin),
          row.unstable ? "unstable" : "stable",
        ].join("  "));
      }

      console.log("\\nLoss impact diagnostics");
      for (const row of lossMobilityImpacts().sort((left, right) =>
        Math.abs(right.totalMobilityDelta) - Math.abs(left.totalMobilityDelta)
      ).slice(0, 8)) {
        console.log([
          pieceLabel(row.removedPieceId).padEnd(12),
          "totalΔ=" + row.totalMobilityDelta,
          "ownerΔ=" + row.ownerMobilityDelta,
          "enemyΔ=" + row.opponentMobilityDelta,
          "signΔ=" + row.signChanges,
          "unstableΔ=" + row.unstableDelta,
          "ownKingΔ=" + round(row.ownKingMarginDelta),
          "enemyKingΔ=" + round(row.enemyKingMarginDelta),
        ].join("  "));
      }

      console.log("\\nMost volatile legal moves from current tuning");
      for (const row of snapshot.moveConsequences.topVolatileMoves) {
        console.log([
          pieceLabel(row.pieceId).padEnd(12),
          "to=" + destinationLabel(row.destination),
          "signΔ=" + row.fieldSignChanges,
          "l1Δ=" + round(row.fieldL1Delta),
          "ownMobΔ=" + row.actingMobilityDelta,
          "enemyMobΔ=" + row.enemyMobilityDelta,
          "unstableΔ=" + row.unstableDelta,
          "enemyKingΔ=" + round(row.enemyKingMarginDelta),
        ].join("  "));
      }

      console.log("\\nTrap-shaped legal moves from current tuning");
      for (const row of snapshot.moveConsequences.topTrapMoves) {
        console.log([
          pieceLabel(row.pieceId).padEnd(12),
          "to=" + destinationLabel(row.destination),
          "trap=" + round(row.trapScore),
          "apparentSafety=" + round(row.apparentSafetyScore),
          "enemyMobΔ=" + row.enemyMobilityDelta,
          "enemyKingΔ=" + round(row.enemyKingMarginDelta),
          "unstableΔ=" + row.unstableDelta,
          "signΔ=" + row.fieldSignChanges,
        ].join("  "));
      }

      console.log("\\nLure-trap legal moves from current tuning");
      for (const row of snapshot.moveConsequences.topLureTrapMoves) {
        console.log([
          pieceLabel(row.pieceId).padEnd(12),
          "to=" + destinationLabel(row.destination),
          "lureTrap=" + round(row.lureTrapScore),
          "apparentSafety=" + round(row.apparentSafetyScore),
          "enemyMobΔ=" + row.enemyMobilityDelta,
          "enemyKingΔ=" + round(row.enemyKingMarginDelta),
          "unstableΔ=" + row.unstableDelta,
          "signΔ=" + row.fieldSignChanges,
        ].join("  "));
      }

      console.log("\\nWave scale search");
      const scaleOptions = {
        pawn: [
          { friendly: 4, hostile: 1 },
          { friendly: 3, hostile: 1.5 },
          { friendly: 3, hostile: 2 },
        ],
        rook: [
          { friendly: 3, hostile: 1 },
          { friendly: 2.5, hostile: 1.5 },
          { friendly: 2, hostile: 1.5 },
          { friendly: 2, hostile: 0 },
        ],
        spy: [
          { friendly: 3, hostile: 0 },
          { friendly: 2.5, hostile: 0.5 },
          { friendly: 2, hostile: 1 },
        ],
        king: [
          { friendly: 4, hostile: 2 },
          { friendly: 3, hostile: 2 },
          { friendly: 3, hostile: 2.5 },
          { friendly: 3, hostile: 0 },
        ],
      };
      const currentResult = evaluateWaveScaleSet(DEFAULT_WAVE_SCALES);
      console.log("current  " + resultLine(currentResult));
      console.log("         " + deadProfileLine(currentResult));
      const candidates = searchWaveScaleOptions(scaleOptions, 10);
      candidates.forEach((candidate, index) => {
        console.log(String(index + 1).padStart(2) + ".       " + resultLine(candidate));
        console.log("         " + deadProfileLine(candidate));
      });

      console.log("\\nDefault component search");
      const currentDefaults = evaluateDefaultComponentSet(DEFAULT_COMPONENTS);
      console.log([
        "current",
        "score=" + round(currentDefaults.score),
        "moves=" + currentDefaults.openingMoves.blue + "/" + currentDefaults.openingMoves.red,
        "maxRatio=" + round(currentDefaults.selectedProfileMaxPolarityRatio),
        "minMob=" + round(currentDefaults.selectedProfileMinMobility),
        componentsLabel(currentDefaults.components),
      ].join("  "));
      searchDefaultComponentSets(10).forEach((candidate, index) => {
        console.log([
          String(index + 1).padStart(2) + ".",
          "score=" + round(candidate.score),
          "moves=" + candidate.openingMoves.blue + "/" + candidate.openingMoves.red,
          "maxRatio=" + round(candidate.selectedProfileMaxPolarityRatio),
          "minMob=" + round(candidate.selectedProfileMinMobility),
          componentsLabel(candidate.components),
        ].join("  "));
      });

      console.log("\\nHome energy search");
      const currentHomeEnergy = evaluateHomeEnergySet(DEFAULT_HOME_ENERGY);
      console.log([
        "current",
        "score=" + round(currentHomeEnergy.score),
        "unstable=" + currentHomeEnergy.unstablePieces,
        "minMargin=" + round(currentHomeEnergy.minPieceMargin),
        "avgMargin=" + round(currentHomeEnergy.averagePieceMargin),
        "bigHat=" + round(currentHomeEnergy.bigHatMargins.blue) + "/" + round(currentHomeEnergy.bigHatMargins.red),
        homeEnergyLabel(currentHomeEnergy.homeEnergy),
      ].join("  "));
      const homeEnergyOptions = {
        pawn: [0, 0.25, 0.5],
        rook: [0, 0.25, 0.5],
        spy: [0, 0.5, 1],
        king: [0, 0.25, 0.5, 1],
      };
      searchHomeEnergyOptions(homeEnergyOptions, 10).forEach((candidate, index) => {
        console.log([
          String(index + 1).padStart(2) + ".",
          "score=" + round(candidate.score),
          "unstable=" + candidate.unstablePieces,
          "minMargin=" + round(candidate.minPieceMargin),
          "avgMargin=" + round(candidate.averagePieceMargin),
          "bigHat=" + round(candidate.bigHatMargins.blue) + "/" + round(candidate.bigHatMargins.red),
          homeEnergyLabel(candidate.homeEnergy),
        ].join("  "));
      });

      console.log("\\nPattern variant search");
      const variantTargets = [
        { pieceType: "rook", componentIndex: 0 },
        { pieceType: "rook", componentIndex: 1 },
        { pieceType: "king", componentIndex: 0 },
        { pieceType: "king", componentIndex: 2 },
      ];
      searchDefinitionVariants(variantTargets, 12).forEach((candidate, index) => {
        console.log([
          String(index + 1).padStart(2) + ".",
          candidate.pieceType + " C" + (candidate.componentIndex + 1),
          candidate.definition.name,
          "score=" + round(candidate.score),
          "maxRatio=" + round(candidate.maxPolarityRatio),
          "dead=" + candidate.deadProfiles.length,
          "minMob=" + round(candidate.minAverageMoves),
          "minFirst=" + round(candidate.minAverageFirstSteps),
          "net=" + round(candidate.componentNetMass),
          "balance=" + round(candidate.componentSignBalance),
          "adj=" + candidate.adjacentCounts.positive + "/" + candidate.adjacentCounts.negative + "/" + candidate.adjacentCounts.zero,
        ].join("  "));
      });

      console.log("\\nCoupled pattern variant search");
      const coupledTargets = [
        [
          { pieceType: "rook", componentIndex: 0 },
          { pieceType: "rook", componentIndex: 1 },
        ],
        [
          { pieceType: "king", componentIndex: 0 },
          { pieceType: "king", componentIndex: 2 },
        ],
      ];
      searchDefinitionSetVariants(coupledTargets, 12).forEach((candidate, index) => {
        const names = candidate.replacements.map((replacement) =>
          replacement.pieceType + " C" + (replacement.componentIndex + 1) + "=" + replacement.definition.name
        ).join(" | ");
        const dead = candidate.deadProfiles.length === 0
          ? "none"
          : candidate.deadProfiles.map((row) => row.pieceType + profileLabel(row.profile) + ":" + round(row.averageMoves)).join(",");
        const adj = candidate.adjacentSummary.map((row) =>
          row.pieceType + "C" + (row.componentIndex + 1) + ":" + row.positive + "/" + row.negative + "/" + row.zero
        ).join(" ");
        console.log([
          String(index + 1).padStart(2) + ".",
          "score=" + round(candidate.score),
          "maxRatio=" + round(candidate.maxPolarityRatio),
          "dead=" + dead,
          "minMob=" + round(candidate.minAverageMoves),
          "minFirst=" + round(candidate.minAverageFirstSteps),
          "adj=" + adj,
          names,
        ].join("  "));
      });

      console.log("\\nCombined candidate search");
      const mixedDefaults = {
        pawn: [1],
        rook: [1, -1],
        spy: [0, 0, -1],
        king: [0, -1, 1],
      };
      const softenedScales = {
        pawn: { friendly: 3, hostile: 2 },
        rook: { friendly: 2, hostile: 1.5 },
        spy: { friendly: 2, hostile: 1 },
        king: { friendly: 3, hostile: 2.5 },
      };
      const neutralFirstRingScales = {
        pawn: { friendly: 4, hostile: 1 },
        rook: { friendly: 2, hostile: 0 },
        spy: { friendly: 2, hostile: 1 },
        king: { friendly: 3, hostile: 0 },
      };
      const moderateHomeBoost = { pawn: 0.5, rook: 2, spy: 0.5, king: 2 };
      const highHomeBoost = { pawn: 1, rook: 5, spy: 1, king: 6 };
      const rookNeutralFirstRing = [
        { pieceType: "rook", componentIndex: 0, definition: ringVariant("rook", 0, [0, 0, 1, -1]) },
        { pieceType: "rook", componentIndex: 1, definition: ringVariant("rook", 1, [0, 0, 1, -1]) },
      ];
      const userTowerAction = [
        { pieceType: "rook", componentIndex: 0, definition: ringVariant("rook", 0, [0, 0, 1, -1]) },
        { pieceType: "rook", componentIndex: 1, definition: ringVariant("rook", 1, [0, 1, 0, -1]) },
      ];
      const userBigHatAstigmatism = [
        { pieceType: "king", componentIndex: 2, definition: presetVariant("king", 2, "astigmatism") },
      ];
      const kingAstigmatic = [
        { pieceType: "king", componentIndex: 0, definition: ringVariant("king", 0, [0, 1, -1, -1, 0, 1]) },
        { pieceType: "king", componentIndex: 2, definition: presetVariant("king", 2, "astigmatism") },
      ];
      const combinedCandidates = [
        { name: "current" },
        { name: "mixed-defaults", components: mixedDefaults },
        { name: "softened-scales", scales: softenedScales },
        { name: "zero-hostile-diagnostic", scales: neutralFirstRingScales },
        { name: "rook-pattern", replacements: rookNeutralFirstRing },
        { name: "king-pattern", replacements: kingAstigmatic },
        { name: "user-tower[0+-][+0-]", replacements: userTowerAction },
        { name: "user-big-hat-c3-astigmatism", replacements: userBigHatAstigmatism },
        { name: "user-tower+big-hat-c3", replacements: [...userTowerAction, ...userBigHatAstigmatism] },
        { name: "rook+king-pattern", replacements: [...rookNeutralFirstRing, ...kingAstigmatic] },
        { name: "patterns+mixed-defaults", replacements: [...rookNeutralFirstRing, ...kingAstigmatic], components: mixedDefaults },
        { name: "patterns+soft-scales", replacements: [...rookNeutralFirstRing, ...kingAstigmatic], scales: softenedScales },
        { name: "patterns+moderate-home", replacements: [...rookNeutralFirstRing, ...kingAstigmatic], homeEnergy: moderateHomeBoost },
        { name: "patterns+high-home", replacements: [...rookNeutralFirstRing, ...kingAstigmatic], homeEnergy: highHomeBoost },
        { name: "patterns+soft-scales+mixed-defaults", replacements: [...rookNeutralFirstRing, ...kingAstigmatic], scales: softenedScales, components: mixedDefaults },
        { name: "patterns+soft-scales+high-home", replacements: [...rookNeutralFirstRing, ...kingAstigmatic], scales: softenedScales, homeEnergy: highHomeBoost },
      ];
      searchCombinedParameterCandidates(combinedCandidates, combinedCandidates.length).forEach((candidate, index) => {
        const replacements = candidate.replacements.length === 0
          ? "none"
          : candidate.replacements.map((replacement) => replacement.pieceType + "C" + (replacement.componentIndex + 1)).join(",");
        console.log([
          String(index + 1).padStart(2) + ".",
          candidate.name,
          "score=" + round(candidate.score),
          "dead=" + candidate.wave.deadProfiles.length,
          "maxRatio=" + round(candidate.wave.maxPolarityRatio),
          "minMob=" + round(candidate.wave.minMobility),
          "spread=" + round(candidate.wave.mobilitySpread),
          "defaultRatio=" + round(candidate.defaults.selectedProfileMaxPolarityRatio),
          "moves=" + candidate.defaults.openingMoves.blue + "/" + candidate.defaults.openingMoves.red,
          "homeMin=" + round(candidate.home.minPieceMargin),
          "repl=" + replacements,
        ].join("  "));
      });

      console.log("\\nPairwise combo outliers" + (full ? " (full board)" : " (sampled positions; pass --full for exhaustive board)"));
      const outliers = findComboOutliers(undefined, 5, full ? {} : { positions: sampledPositions });
      for (const [name, rows] of Object.entries(outliers)) {
        console.log("\\n" + name);
        for (const row of rows) {
          console.log([
            row.first.pieceType + profileLabel(row.first.profile) + "@" + row.first.position.x + "," + row.first.position.y,
            row.second.pieceType + profileLabel(row.second.profile) + "@" + row.second.position.x + "," + row.second.position.y,
            "l1=" + round(row.metrics.l1),
            "net=" + round(row.metrics.netMass),
            "edges=" + row.metrics.edgeCount,
            "reinforce=" + round(row.metrics.sameSignReinforcement),
            "cancel=" + round(row.metrics.cancellation),
            "max=" + round(row.metrics.maxAbs),
          ].join("  "));
        }
      }
    `,
    resolveDir: process.cwd(),
    sourcefile: "mechanics-report.ts",
    loader: "ts",
  },
  bundle: true,
  format: "esm",
  platform: "node",
  outfile,
  logLevel: "silent",
});

await import(pathToFileURL(outfile).href + `?cache=${Date.now()}`);
