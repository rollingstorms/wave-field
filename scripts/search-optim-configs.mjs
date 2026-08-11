import { build } from "esbuild";
import { pathToFileURL } from "node:url";

const iterations = Number(process.env.OPTIM_ITERATIONS ?? 3);
const population = Number(process.env.OPTIM_POPULATION ?? 10);
const eliteCount = Number(process.env.OPTIM_ELITES ?? 4);
const games = Number(process.env.GAMES ?? 3);
const maxHalfTurns = Number(process.env.MAX_HALF_TURNS ?? 90);
const timeBudgetMs = Number(process.env.AI_BUDGET_MS ?? 25);
const variety = Number(process.env.AI_VARIETY ?? 0.45);
const seed = Number(process.env.OPTIM_SEED ?? 1701);
const outfile = "/tmp/wave-field-optim-search.mjs";

await build({
  stdin: {
    contents: `
      import { candidateDefinitionVariants, complexitySnapshot, evaluateCombinedParameterCandidate } from "./src/game/mechanicsAnalysis.ts";
      import { DEFAULT_COMPONENTS, cloneDefinitions } from "./src/field/componentDefinitions.ts";
      import { DEFAULT_HOME_ENERGY, DEFAULT_WAVE_SCALES } from "./src/game/constants.ts";
      import { evaluateField } from "./src/field/evaluateField.ts";
      import { projectFieldValue } from "./src/field/projection.ts";
      import { createInitialState } from "./src/game/initialState.ts";
      import { getLegalMoves } from "./src/game/movement.ts";
      import { playHeuristicTurn } from "./src/game/ai.ts";
      import { activationOrdersForPlayers } from "./src/game/tuning.ts";
      import { getUnstablePieces, isKingUnprotected } from "./src/game/victory.ts";

      const iterations = ${JSON.stringify(iterations)};
      const population = ${JSON.stringify(population)};
      const eliteCount = ${JSON.stringify(eliteCount)};
      const games = ${JSON.stringify(games)};
      const maxHalfTurns = ${JSON.stringify(maxHalfTurns)};
      const timeBudgetMs = ${JSON.stringify(timeBudgetMs)};
      const variety = ${JSON.stringify(variety)};
      let randomState = ${JSON.stringify(seed)} >>> 0;
      const pieceTypes = ["pawn", "rook", "spy", "king"];
      const materialValue = { pawn: 2, rook: 4, spy: 3, king: 0 };

      function random() {
        randomState = Math.imul(randomState ^ (randomState >>> 15), 1 | randomState);
        randomState ^= randomState + Math.imul(randomState ^ (randomState >>> 7), 61 | randomState);
        return ((randomState ^ (randomState >>> 14)) >>> 0) / 4294967296;
      }

      function pick(values) {
        return values[Math.floor(random() * values.length)];
      }

      function round(value) {
        return Math.round(value * 1000) / 1000;
      }

      function profileLabel(profile) {
        return "[" + profile.map((value) => value > 0 ? "+" : value < 0 ? "-" : "0").join(" ") + "]";
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

      function ringReplacement(pieceType, componentIndex, values, name) {
        const definitions = cloneDefinitions();
        const current = definitions[pieceType][componentIndex];
        return {
          pieceType,
          componentIndex,
          definition: {
            ...current,
            name,
            kind: "ring",
            geometry: "chebyshev",
            ringValues: values,
            repeat: true,
          },
        };
      }

      function presetReplacement(pieceType, componentIndex, preset, name) {
        const definitions = cloneDefinitions();
        const current = definitions[pieceType][componentIndex];
        return {
          pieceType,
          componentIndex,
          definition: {
            ...current,
            name,
            kind: "preset",
            preset,
          },
        };
      }

      const knobs = {
        rookC1: [
          ringVariant("rook", 0, [0, 0, 1, -1]),
          ringVariant("rook", 0, [0, 1, 0, -1]),
          ringVariant("rook", 0, [0, 1, -1, -1]),
          ringVariant("rook", 0, [0, 1, -1, 1]),
        ],
        rookC2: [
          ringVariant("rook", 1, [0, 0, 1, -1]),
          ringVariant("rook", 1, [0, 1, 0, -1]),
          ringVariant("rook", 1, [0, 1, -1, -1]),
          ringVariant("rook", 1, [0, 1, -1, 1]),
        ],
        kingC1: [
          ringReplacement("king", 0, [1, 1, -1, -1, 1], "Big Hat c1 rings").definition,
          ringReplacement("king", 0, [1, 1, -1, 1], "Yesterday Big Hat c1 rings").definition,
          ringReplacement("king", 0, [0, 1, -1, -1, 0], "Optim Big Hat pressure rings").definition,
          ringReplacement("king", 0, [0, 1, -1, 1, 0], "Optim Big Hat alternating rings").definition,
        ],
        kingC3: [
          presetVariant("king", 2, "block-checker"),
          presetVariant("king", 2, "astigmatism"),
          presetVariant("king", 2, "checkerboard"),
          presetVariant("king", 2, "axis-favor"),
          presetVariant("king", 2, "diagonal-favor"),
        ],
        spyC2: [
          presetVariant("spy", 1, "diamond-core"),
          presetVariant("spy", 1, "axis-favor"),
          presetVariant("spy", 1, "diagonal-favor"),
          presetVariant("spy", 1, "astigmatism"),
        ],
        pawnScale: [
          { friendly: 4, hostile: 1 },
          { friendly: 3, hostile: 1.5 },
          { friendly: 3, hostile: 2 },
        ],
        rookScale: [
          { friendly: 3, hostile: 1 },
          { friendly: 2.5, hostile: 1 },
          { friendly: 2, hostile: 0 },
          { friendly: 2, hostile: 1.5 },
        ],
        spyScale: [
          { friendly: 3, hostile: 0 },
          { friendly: 2.5, hostile: 0.5 },
          { friendly: 2, hostile: 1 },
        ],
        kingScale: [
          { friendly: 4, hostile: 2 },
          { friendly: 3, hostile: 0 },
          { friendly: 3, hostile: 2 },
          { friendly: 3, hostile: 2.5 },
        ],
        home: [
          { pawn: 0, rook: 0, spy: 0.5, king: 0 },
          { pawn: 0, rook: 0.5, spy: 0.5, king: 0.5 },
          { pawn: 0, rook: 1, spy: 0.5, king: 1 },
          { pawn: 0.25, rook: 0.5, spy: 0.25, king: 0.5 },
        ],
        components: [
          DEFAULT_COMPONENTS,
          { pawn: [1], rook: [-1, 1], spy: [1, 0], king: [1, 1] },
          { pawn: [1], rook: [-1, 1], spy: [0, 1], king: [-1, 1] },
          { pawn: [1], rook: [-1, 1], spy: [0, -1], king: [1, -1] },
          { pawn: [1], rook: [1, 1], spy: [0, 1], king: [-1, 1] },
        ],
      };

      function candidateFromGenes(genes, name) {
        return {
          name,
          replacements: [
            { pieceType: "rook", componentIndex: 0, definition: genes.rookC1 },
            { pieceType: "rook", componentIndex: 1, definition: genes.rookC2 },
            { pieceType: "king", componentIndex: 0, definition: genes.kingC1 },
            { pieceType: "king", componentIndex: 2, definition: genes.kingC3 },
            { pieceType: "spy", componentIndex: 1, definition: genes.spyC2 },
          ],
          scales: {
            pawn: genes.pawnScale,
            rook: genes.rookScale,
            spy: genes.spyScale,
            king: genes.kingScale,
          },
          homeEnergy: genes.home,
          components: genes.components,
        };
      }

      function randomGenes() {
        return Object.fromEntries(Object.entries(knobs).map(([key, values]) => [key, structuredClone(pick(values))]));
      }

      function baselineGenes() {
        return {
          rookC1: structuredClone(ringVariant("rook", 0, [0, 0, 1, -1])),
          rookC2: structuredClone(ringVariant("rook", 1, [0, 1, 0, -1])),
          kingC1: structuredClone(ringReplacement("king", 0, [1, 1, -1, -1, 1], "Big Hat c1 rings").definition),
          kingC3: structuredClone(presetVariant("king", 2, "astigmatism")),
          spyC2: structuredClone(presetVariant("spy", 1, "diagonal-favor")),
          pawnScale: structuredClone(DEFAULT_WAVE_SCALES.pawn),
          rookScale: structuredClone(DEFAULT_WAVE_SCALES.rook),
          spyScale: structuredClone(DEFAULT_WAVE_SCALES.spy),
          kingScale: structuredClone(DEFAULT_WAVE_SCALES.king),
          home: structuredClone(DEFAULT_HOME_ENERGY),
          components: structuredClone(DEFAULT_COMPONENTS),
        };
      }

      function mutateGenes(parent) {
        const genes = structuredClone(parent);
        for (const [key, values] of Object.entries(knobs)) {
          if (random() < 0.32) genes[key] = structuredClone(pick(values));
        }
        return genes;
      }

      function definitionsWith(replacements) {
        const definitions = cloneDefinitions();
        for (const replacement of replacements) {
          if (replacement.definition.kind === "ring" && !Array.isArray(replacement.definition.ringValues)) {
            throw new Error("Invalid ring replacement " + replacement.pieceType + " C" + (replacement.componentIndex + 1) + " " + JSON.stringify(replacement.definition));
          }
          if (replacement.definition.kind === "preset" && typeof replacement.definition.preset !== "string") {
            throw new Error("Invalid preset replacement " + replacement.pieceType + " C" + (replacement.componentIndex + 1) + " " + JSON.stringify(replacement.definition));
          }
          definitions[replacement.pieceType][replacement.componentIndex] = structuredClone(replacement.definition);
        }
        return definitions;
      }

      function createConfigState(candidate) {
        const state = createInitialState(
          structuredClone(candidate.components),
          definitionsWith(candidate.replacements),
          structuredClone(candidate.scales),
          structuredClone(candidate.homeEnergy),
        );
        state.components = {
          blue: structuredClone(candidate.components),
          red: structuredClone(candidate.components),
        };
        state.activationOrders = activationOrdersForPlayers(state.components);
        return state;
      }

      function validateCandidate(candidate) {
        for (const replacement of candidate.replacements) {
          if (replacement.definition.kind === "ring" && !Array.isArray(replacement.definition.ringValues)) {
            throw new Error("Invalid ring replacement " + replacement.pieceType + " C" + (replacement.componentIndex + 1) + " " + JSON.stringify(replacement.definition));
          }
          if (replacement.definition.kind === "preset" && typeof replacement.definition.preset !== "string") {
            throw new Error("Invalid preset replacement " + replacement.pieceType + " C" + (replacement.componentIndex + 1) + " " + JSON.stringify(replacement.definition));
          }
        }
      }

      function validateStateDefinitions(state, candidateName) {
        for (const pieceType of pieceTypes) {
          state.definitions[pieceType].forEach((definition, index) => {
            if (definition.kind === "ring" && !Array.isArray(definition.ringValues)) {
              throw new Error("Invalid state ring " + candidateName + " " + pieceType + " C" + (index + 1) + " " + JSON.stringify(definition));
            }
          });
        }
      }

      function stateKey(state) {
        const pieces = [...state.pieces]
          .sort((left, right) => left.id.localeCompare(right.id))
          .map((piece) => piece.id + ":" + piece.position.x + "," + piece.position.y)
          .join("|");
        const components = ["blue", "red"].map((side) =>
          side + ":" + pieceTypes.map((pieceType) => state.components[side][pieceType].join(",")).join("/")
        ).join("|");
        return state.currentPlayer + "|" + pieces + "|" + components;
      }

      function pieceMaterial(state, player) {
        return state.pieces
          .filter((piece) => piece.owner === player)
          .reduce((total, piece) => total + materialValue[piece.type], 0);
      }

      function legalMoveCount(state, player = state.currentPlayer) {
        const field = evaluateField(state);
        return state.pieces
          .filter((piece) => piece.owner === player)
          .reduce((total, piece) => total + getLegalMoves(piece.id, state, field).length, 0);
      }

      function openingTerritoryConstraint(state) {
        const field = evaluateField(state);
        let mismatches = 0;
        for (let y = 0; y < field.length; y += 1) {
          for (let x = 0; x < field[y].length; x += 1) {
            const territory = projectFieldValue(field[y][x]);
            const expected = y < 3 ? "blue" : y === 3 ? "neutral" : "red";
            if (territory !== expected) mismatches += 1;
          }
        }
        return { ok: mismatches === 0, mismatches };
      }

      function messageHasCheck(message) {
        return /Big Hat is in check/.test(message);
      }

      function rolloutMetrics(candidate) {
        const aggregate = {
          redWins: 0,
          blueWins: 0,
          limitGames: 0,
          halfTurns: 0,
          checks: 0,
          checkDeliveries: 0,
          rescueSuccesses: 0,
          noRescueLosses: 0,
          terminalNoRescue: 0,
          terminalNoMove: 0,
          materialLosses: 0,
          unstableTurns: 0,
          repeatTurns: 0,
          comebackWins: 0,
          legalMoves: [],
        };

        for (let game = 0; game < games; game += 1) {
          let state = createConfigState(candidate);
          let halfTurns = 0;
          let finishedByLimit = true;
          const seen = new Map([[stateKey(state), 1]]);
          const worstDeficit = { red: 0, blue: 0 };

          while (state.status === "playing" && halfTurns < maxHalfTurns) {
            const before = state;
            const player = state.currentPlayer;
            const field = evaluateField(before);
            const inCheck = isKingUnprotected(player, before, field) || messageHasCheck(before.message);
            const beforeRedMaterial = pieceMaterial(before, "red");
            const beforeBlueMaterial = pieceMaterial(before, "blue");
            worstDeficit.red = Math.min(worstDeficit.red, beforeRedMaterial - beforeBlueMaterial);
            worstDeficit.blue = Math.min(worstDeficit.blue, beforeBlueMaterial - beforeRedMaterial);
            aggregate.legalMoves.push(legalMoveCount(before, player));
            aggregate.unstableTurns += getUnstablePieces(player, before, field).length;
            if (inCheck) aggregate.checks += 1;

            state = playHeuristicTurn(before, player, {
              seed: game * 4099 + halfTurns * 97 + (player === "red" ? 13 : 29),
              variety,
              timeBudgetMs,
            });

            const afterRedMaterial = pieceMaterial(state, "red");
            const afterBlueMaterial = pieceMaterial(state, "blue");
            aggregate.materialLosses += Math.max(0, beforeRedMaterial - afterRedMaterial) + Math.max(0, beforeBlueMaterial - afterBlueMaterial);
            if (inCheck) {
              if (state.status === "playing") aggregate.rescueSuccesses += 1;
              else if (/no legal rescue|resigned while in check/i.test(state.message)) aggregate.noRescueLosses += 1;
            }
            if (messageHasCheck(state.message)) aggregate.checkDeliveries += 1;

            const key = stateKey(state);
            const nextCount = (seen.get(key) ?? 0) + 1;
            if (nextCount > 1) aggregate.repeatTurns += 1;
            seen.set(key, nextCount);
            halfTurns += 1;
          }

          if (state.status !== "playing") finishedByLimit = false;
          aggregate.halfTurns += halfTurns;
          if (state.status === "red-won") {
            aggregate.redWins += 1;
            if (worstDeficit.red <= -2) aggregate.comebackWins += 1;
          }
          if (state.status === "blue-won") {
            aggregate.blueWins += 1;
            if (worstDeficit.blue <= -2) aggregate.comebackWins += 1;
          }
          if (finishedByLimit) aggregate.limitGames += 1;
          if (/no legal rescue/i.test(state.message)) aggregate.terminalNoRescue += 1;
          if (/no legal move/i.test(state.message)) aggregate.terminalNoMove += 1;
        }

        const legal = aggregate.legalMoves;
        const avgLegalMoves = legal.reduce((total, value) => total + value, 0) / Math.max(1, legal.length);
        return {
          terminalRate: (aggregate.redWins + aggregate.blueWins) / games,
          limitRate: aggregate.limitGames / games,
          avgHalfTurns: aggregate.halfTurns / games,
          checksPerGame: aggregate.checks / games,
          checkDeliveriesPerGame: aggregate.checkDeliveries / games,
          rescueRate: aggregate.checks === 0 ? 0 : aggregate.rescueSuccesses / aggregate.checks,
          noRescueLosses: aggregate.noRescueLosses,
          terminalNoRescue: aggregate.terminalNoRescue,
          terminalNoMove: aggregate.terminalNoMove,
          materialLossesPerGame: aggregate.materialLosses / games,
          unstableTurnsPerGame: aggregate.unstableTurns / games,
          repeatTurnsPerGame: aggregate.repeatTurns / games,
          comebackRate: aggregate.comebackWins / Math.max(1, aggregate.redWins + aggregate.blueWins),
          avgLegalMoves,
          redWins: aggregate.redWins,
          blueWins: aggregate.blueWins,
          limitGames: aggregate.limitGames,
        };
      }

      function scoreCandidate(candidate) {
        validateCandidate(candidate);
        const state = createConfigState(candidate);
        validateStateDefinitions(state, candidate.name);
        const openingTerritory = openingTerritoryConstraint(state);
        const staticResult = evaluateCombinedParameterCandidate(candidate, state);
        const snapshot = complexitySnapshot(state, 0);
        const dynamic = openingTerritory.ok
          ? rolloutMetrics(candidate)
          : {
            terminalRate: 0,
            limitRate: 1,
            avgHalfTurns: maxHalfTurns,
            checksPerGame: 0,
            checkDeliveriesPerGame: 0,
            rescueRate: 1,
            noRescueLosses: 0,
            terminalNoRescue: 0,
            terminalNoMove: 0,
            materialLossesPerGame: 0,
            unstableTurnsPerGame: 0,
            repeatTurnsPerGame: 0,
            comebackRate: 0,
            avgLegalMoves: 0,
            redWins: 0,
            blueWins: 0,
            limitGames: games,
          };
        const move = snapshot.moveConsequences;
        const terminalReward = dynamic.terminalRate * 80;
        const lengthReward = 34 - Math.abs(dynamic.avgHalfTurns - 46) * 0.75;
        const checkReward = 22 - Math.abs(dynamic.checksPerGame - 2.5) * 7;
        const lossReward = 20 - Math.abs(dynamic.materialLossesPerGame - 2.25) * 8;
        const underdogReward = dynamic.comebackRate * 45;
        const fieldReward = Math.min(24, move.averageFieldL1Delta * 0.55) + Math.min(16, move.averageSignChanges * 2.5);
        const rescuePressureReward = dynamic.noRescueLosses * 18
          + dynamic.terminalNoRescue * 30
          + Math.max(0, 0.8 - dynamic.rescueRate) * 42;
        const excessiveTrapPenalty = Math.max(0, dynamic.checksPerGame - 5) * 12;
        const standoffPenalty = dynamic.limitRate * 55
          + dynamic.repeatTurnsPerGame * 14
          + Math.max(0, dynamic.rescueRate - 0.75) * 90
          + excessiveTrapPenalty;
        const staticPenalty = openingTerritory.mismatches * 1000
          + staticResult.wave.deadProfiles.length * 20
          + Math.max(0, staticResult.wave.maxPolarityRatio - 4) * 7
          + Math.max(0, 1 - staticResult.home.minPieceMargin) * 24
          + Math.max(0, dynamic.avgLegalMoves - 48) * 1.5
          + Math.max(0, 24 - dynamic.avgLegalMoves) * 2;
        const score = terminalReward + lengthReward + checkReward + lossReward + underdogReward + fieldReward + rescuePressureReward - standoffPenalty - staticPenalty;
        return { candidate, staticResult, snapshot, dynamic, openingTerritory, score };
      }

      function summarize(result) {
        const candidate = result.candidate;
        const replacements = Object.fromEntries(candidate.replacements.map((replacement) => [
          replacement.pieceType + "C" + (replacement.componentIndex + 1),
          replacement.definition.kind === "ring"
            ? replacement.definition.ringValues.join("/")
            : replacement.definition.preset,
        ]));
        return [
          result.candidate.name.padEnd(18),
          "score=" + round(result.score),
          "term=" + round(result.dynamic.terminalRate),
          "limit=" + round(result.dynamic.limitRate),
          "halfTurns=" + round(result.dynamic.avgHalfTurns),
          "checks=" + round(result.dynamic.checksPerGame),
          "rescue=" + round(result.dynamic.rescueRate),
          "failedRescue=" + result.dynamic.noRescueLosses,
          "terminalNoRescue=" + result.dynamic.terminalNoRescue,
          "losses=" + round(result.dynamic.materialLossesPerGame),
          "unstable=" + round(result.dynamic.unstableTurnsPerGame),
          "repeat=" + round(result.dynamic.repeatTurnsPerGame),
          "comeback=" + round(result.dynamic.comebackRate),
          "moves=" + round(result.dynamic.avgLegalMoves),
          "openingRows=" + (result.openingTerritory.ok ? "ok" : "bad:" + result.openingTerritory.mismatches),
          "dead=" + result.staticResult.wave.deadProfiles.length,
          "ratio=" + round(result.staticResult.wave.maxPolarityRatio),
          "homeMin=" + round(result.staticResult.home.minPieceMargin),
          "defs=" + JSON.stringify(replacements),
          "scales=" + JSON.stringify(candidate.scales),
          "home=" + JSON.stringify(candidate.homeEnergy),
          "components=" + pieceTypes.map((pieceType) => pieceType + profileLabel(candidate.components[pieceType])).join(" "),
        ].join("  ");
      }

      console.log("Optim config search");
      console.log("iterations=" + iterations + " population=" + population + " elites=" + eliteCount + " games=" + games + " maxHalfTurns=" + maxHalfTurns + " budgetMs=" + timeBudgetMs + " variety=" + variety);

      let elites = [];
      const allResults = [];
      for (let iteration = 0; iteration < iterations; iteration += 1) {
        const generation = [];
        for (let index = 0; index < population; index += 1) {
          const genes = iteration === 0 && index === 0
            ? baselineGenes()
            : elites.length > 0 && random() < 0.7
              ? mutateGenes(pick(elites).genes)
              : randomGenes();
          const candidate = candidateFromGenes(genes, "iter" + iteration + "-" + index);
          const result = { ...scoreCandidate(candidate), genes };
          generation.push(result);
          allResults.push(result);
        }
        generation.sort((left, right) => right.score - left.score);
        elites = [...elites, ...generation].sort((left, right) => right.score - left.score).slice(0, eliteCount);
        console.log("\\nIteration " + (iteration + 1));
        for (const result of generation.slice(0, Math.min(5, generation.length))) console.log(summarize(result));
      }

      console.log("\\nBest candidates");
      for (const result of [...allResults].sort((left, right) => right.score - left.score).slice(0, 10)) {
        console.log(summarize(result));
      }
    `,
    resolveDir: process.cwd(),
    sourcefile: "optim-search.ts",
    loader: "ts",
  },
  bundle: true,
  format: "esm",
  platform: "node",
  outfile,
  logLevel: "silent",
});

await import(pathToFileURL(outfile).href + `?cache=${Date.now()}`);
