import { build } from "esbuild";
import { pathToFileURL } from "node:url";

const games = Number(process.env.GAMES ?? 24);
const maxHalfTurns = Number(process.env.MAX_HALF_TURNS ?? 120);
const timeBudgetMs = Number(process.env.AI_BUDGET_MS ?? 80);
const variety = Number(process.env.AI_VARIETY ?? 0.35);
const outfile = "/tmp/wave-field-heuristic-eval.mjs";

await build({
  stdin: {
    contents: `
      import { candidateDefinitionVariants } from "./src/game/mechanicsAnalysis.ts";
      import { DEFAULT_COMPONENTS, cloneDefinitions } from "./src/field/componentDefinitions.ts";
      import { evaluateField } from "./src/field/evaluateField.ts";
      import { createInitialState } from "./src/game/initialState.ts";
      import { getLegalMoves } from "./src/game/movement.ts";
      import { playHeuristicTurn } from "./src/game/ai.ts";
      import { getUnstablePieces, isKingUnprotected } from "./src/game/victory.ts";
      import { activationOrdersForPlayers } from "./src/game/tuning.ts";

      const games = ${JSON.stringify(games)};
      const maxHalfTurns = ${JSON.stringify(maxHalfTurns)};
      const timeBudgetMs = ${JSON.stringify(timeBudgetMs)};
      const variety = ${JSON.stringify(variety)};
      const pieceTypes = ["pawn", "rook", "spy", "king"];

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
        if (current.kind !== "ring") throw new Error(pieceType + " C" + componentIndex + " is not a ring definition");
        return {
          pieceType,
          componentIndex,
          definition: {
            ...current,
            name,
            ringValues: values,
          },
        };
      }

      function definitionsWith(replacements) {
        const definitions = cloneDefinitions();
        for (const replacement of replacements) {
          definitions[replacement.pieceType][replacement.componentIndex] = structuredClone(replacement.definition);
        }
        return definitions;
      }

      function createConfigState(config) {
        const components = structuredClone(config.components ?? DEFAULT_COMPONENTS);
        const state = createInitialState(components, definitionsWith(config.replacements ?? []));
        state.components = {
          blue: structuredClone(components),
          red: structuredClone(components),
        };
        state.activationOrders = activationOrdersForPlayers(state.components);
        return state;
      }

      function legalMoveCount(state, player = state.currentPlayer) {
        const field = evaluateField(state);
        return state.pieces
          .filter((piece) => piece.owner === player)
          .reduce((total, piece) => total + getLegalMoves(piece.id, state, field).length, 0);
      }

      function profileChangeCount(before, after, player) {
        let changes = 0;
        for (const pieceType of pieceTypes) {
          before.components[player][pieceType].forEach((value, index) => {
            if (value !== after.components[player][pieceType][index]) changes += 1;
          });
        }
        return changes;
      }

      function pieceCounts(state) {
        return {
          red: state.pieces.filter((piece) => piece.owner === "red").length,
          blue: state.pieces.filter((piece) => piece.owner === "blue").length,
        };
      }

      function round(value) {
        return Math.round(value * 1000) / 1000;
      }

      function percentile(values, p) {
        if (values.length === 0) return 0;
        const sorted = [...values].sort((a, b) => a - b);
        return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * p))];
      }

      function messageHasCheck(message) {
        return /Big Hat is in check/.test(message);
      }

      function evaluateConfig(config) {
        const aggregate = {
          name: config.name,
          games: 0,
          redWins: 0,
          blueWins: 0,
          maxTurnGames: 0,
          totalHalfTurns: 0,
          checkTurns: 0,
          rescueSuccesses: 0,
          noRescueLosses: 0,
          terminalNoRescueLosses: 0,
          terminalNoMoveLosses: 0,
          checkDeliveries: 0,
          tuneTurns: 0,
          tuneChanges: 0,
          materialLosses: 0,
          unstableTurnTotal: 0,
          legalMoves: [],
          finalReasons: new Map(),
        };

        for (let game = 0; game < games; game += 1) {
          let state = createConfigState(config);
          let halfTurns = 0;
          let finishedByLimit = true;

          while (state.status === "playing" && halfTurns < maxHalfTurns) {
            const before = state;
            const player = state.currentPlayer;
            const field = evaluateField(before);
            const inCheck = isKingUnprotected(player, before, field) || messageHasCheck(before.message);
            const beforeCounts = pieceCounts(before);
            const moves = legalMoveCount(before, player);
            const unstable = getUnstablePieces(player, before, field).length;
            aggregate.legalMoves.push(moves);
            aggregate.unstableTurnTotal += unstable;
            if (inCheck) aggregate.checkTurns += 1;

            state = playHeuristicTurn(before, player, {
              seed: game * 1009 + halfTurns * 37 + (player === "red" ? 17 : 29),
              variety,
              timeBudgetMs,
            });

            const afterCounts = pieceCounts(state);
            aggregate.materialLosses += Math.max(0, beforeCounts.red - afterCounts.red) + Math.max(0, beforeCounts.blue - afterCounts.blue);
            const tuningChanges = profileChangeCount(before, state, player);
            if (tuningChanges > 0) {
              aggregate.tuneTurns += 1;
              aggregate.tuneChanges += tuningChanges;
            }
            if (inCheck) {
              if (state.status === "playing") aggregate.rescueSuccesses += 1;
              else if (/no legal rescue|resigned while in check/i.test(state.message)) aggregate.noRescueLosses += 1;
            }
            if (messageHasCheck(state.message)) aggregate.checkDeliveries += 1;
            halfTurns += 1;
          }

          if (state.status !== "playing") finishedByLimit = false;
          aggregate.games += 1;
          aggregate.totalHalfTurns += halfTurns;
          if (state.status === "red-won") aggregate.redWins += 1;
          if (state.status === "blue-won") aggregate.blueWins += 1;
          if (finishedByLimit) aggregate.maxTurnGames += 1;
          if (/no legal rescue/i.test(state.message)) aggregate.terminalNoRescueLosses += 1;
          if (/no legal move/i.test(state.message)) aggregate.terminalNoMoveLosses += 1;
          aggregate.finalReasons.set(state.status + " " + state.message, (aggregate.finalReasons.get(state.status + " " + state.message) ?? 0) + 1);
        }

        const rescueRate = aggregate.checkTurns === 0 ? 0 : aggregate.rescueSuccesses / aggregate.checkTurns;
        return {
          name: aggregate.name,
          games: aggregate.games,
          redWins: aggregate.redWins,
          blueWins: aggregate.blueWins,
          maxTurnGames: aggregate.maxTurnGames,
          avgHalfTurns: aggregate.totalHalfTurns / aggregate.games,
          checksPerGame: aggregate.checkTurns / aggregate.games,
          checkDeliveriesPerGame: aggregate.checkDeliveries / aggregate.games,
          rescueRate,
          noRescueLosses: aggregate.noRescueLosses,
          terminalNoRescueLosses: aggregate.terminalNoRescueLosses,
          terminalNoMoveLosses: aggregate.terminalNoMoveLosses,
          avgLegalMoves: aggregate.legalMoves.reduce((total, value) => total + value, 0) / Math.max(aggregate.legalMoves.length, 1),
          p25LegalMoves: percentile(aggregate.legalMoves, 0.25),
          p75LegalMoves: percentile(aggregate.legalMoves, 0.75),
          tuneTurnsPerGame: aggregate.tuneTurns / aggregate.games,
          tuneChangesPerGame: aggregate.tuneChanges / aggregate.games,
          materialLossesPerGame: aggregate.materialLosses / aggregate.games,
          unstableTurnsPerGame: aggregate.unstableTurnTotal / aggregate.games,
          finalReasons: [...aggregate.finalReasons.entries()].sort((a, b) => b[1] - a[1]).slice(0, 4),
        };
      }

      const towerUser = [
        { pieceType: "rook", componentIndex: 0, definition: ringVariant("rook", 0, [0, 0, 1, -1]) },
        { pieceType: "rook", componentIndex: 1, definition: ringVariant("rook", 1, [0, 1, 0, -1]) },
      ];
      const bigHatC3Astigmatism = [
        { pieceType: "king", componentIndex: 2, definition: presetVariant("king", 2, "astigmatism") },
      ];
      const yesterdayBigHatC1 = [
        ringReplacement("king", 0, [1, 1, -1, 1], "Yesterday Big Hat c1 rings"),
      ];
      const configs = [
        { name: "yesterday-big-hat-c1", replacements: yesterdayBigHatC1 },
        { name: "today-current" },
        { name: "big-hat-c3-astigmatism", replacements: bigHatC3Astigmatism },
        { name: "tower-[0+-][+0-]", replacements: towerUser },
        { name: "tower+big-hat-c3", replacements: [...towerUser, ...bigHatC3Astigmatism] },
      ];

      console.log("Heuristic self-play eval");
      console.log("games=" + games + " maxHalfTurns=" + maxHalfTurns + " timeBudgetMs=" + timeBudgetMs + " variety=" + variety);
      const rows = configs.map(evaluateConfig);
      for (const row of rows) {
        console.log([
          row.name.padEnd(24),
          "W red/blue/limit=" + row.redWins + "/" + row.blueWins + "/" + row.maxTurnGames,
          "halfTurns=" + round(row.avgHalfTurns),
          "moves=" + round(row.avgLegalMoves) + " [" + row.p25LegalMoves + "-" + row.p75LegalMoves + "]",
          "checks/game=" + round(row.checksPerGame),
          "deliveries/game=" + round(row.checkDeliveriesPerGame),
          "rescueRate=" + round(row.rescueRate),
          "failedRescueTurns=" + row.noRescueLosses,
          "terminalNoRescue=" + row.terminalNoRescueLosses,
          "terminalNoMove=" + row.terminalNoMoveLosses,
          "tuneTurns/game=" + round(row.tuneTurnsPerGame),
          "tuneChanges/game=" + round(row.tuneChangesPerGame),
          "losses/game=" + round(row.materialLossesPerGame),
          "unstable/game=" + round(row.unstableTurnsPerGame),
        ].join("  "));
        for (const [reason, count] of row.finalReasons) console.log("  " + count + "x " + reason);
      }
    `,
    resolveDir: process.cwd(),
    sourcefile: "heuristic-eval.ts",
    loader: "ts",
  },
  bundle: true,
  format: "esm",
  platform: "node",
  outfile,
  logLevel: "silent",
});

await import(pathToFileURL(outfile).href + `?cache=${Date.now()}`);
