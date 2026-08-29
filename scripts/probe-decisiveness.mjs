import { build } from "esbuild";
import { pathToFileURL } from "node:url";

const games = Number(process.env.GAMES ?? 80);
const maxHalfTurns = Number(process.env.MAX_HALF_TURNS ?? 360);
const seed = Number(process.env.SEED ?? 41);
const bots = (process.env.BOTS ?? "random_ish,easy,medium,hard,converter,rescue_first,trap_first,field_control,material_first,chaos")
  .split(",")
  .map((bot) => bot.trim())
  .filter(Boolean);
const checkpoints = (process.env.CHECKPOINTS ?? "30,60,120,180,240,360")
  .split(",")
  .map((value) => Number(value.trim()))
  .filter((value) => Number.isFinite(value) && value > 0)
  .sort((a, b) => a - b);
const outfile = "/tmp/wave-field-decisiveness-probe.mjs";

await build({
  stdin: {
    contents: `
      if (!globalThis.structuredClone) {
        globalThis.structuredClone = (value) => JSON.parse(JSON.stringify(value));
      }

      import { evaluateField, evaluateSignedPieceContribution } from "./src/field/evaluateField.ts";
      import { createInitialState } from "./src/game/initialState.ts";
      import { getLegalMoves } from "./src/game/movement.ts";
      import { applyMove, beginTurn, findClosestPlayableConfiguration, opponent } from "./src/game/rules.ts";
      import { activationOrderForProfile } from "./src/game/tuning.ts";
      import { getUnstablePieces, isKingUnprotected, markInstability } from "./src/game/victory.ts";

      const games = ${JSON.stringify(games)};
      const maxHalfTurns = ${JSON.stringify(maxHalfTurns)};
      const seed = ${JSON.stringify(seed)};
      const bots = ${JSON.stringify(bots)};
      const checkpoints = ${JSON.stringify(checkpoints)};
      const materialValue = { pawn: 2, rook: 4, spy: 3, king: 100 };

      function rngFrom(seedValue) {
        let value = seedValue >>> 0;
        return () => {
          value += 0x6D2B79F5;
          let t = value;
          t = Math.imul(t ^ (t >>> 15), t | 1);
          t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
          return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
      }

      function withPlayer(state, player) {
        return { ...state, currentPlayer: player };
      }

      function material(state, player) {
        return state.pieces
          .filter((piece) => piece.owner === player)
          .reduce((total, piece) => total + materialValue[piece.type], 0);
      }

      function rawKingMoves(state, player) {
        const king = state.pieces.find((piece) => piece.owner === player && piece.type === "king");
        if (!king) return 0;
        return getLegalMoves(king.id, withPlayer(state, player), evaluateField(withPlayer(state, player))).length;
      }

      function pressureAgainst(state, owner, threshold = 1.5) {
        let links = 0;
        for (const target of state.pieces.filter((piece) => piece.owner === owner)) {
          for (const contributor of state.pieces.filter((piece) => piece.owner !== owner)) {
            const signed = evaluateSignedPieceContribution(contributor, target.position, state);
            const hostile = owner === "red" ? -signed : signed;
            if (hostile >= threshold) links += 1;
          }
        }
        return links;
      }

      function legalActions(state, player) {
        const probe = withPlayer(state, player);
        const field = evaluateField(probe);
        return probe.pieces
          .filter((piece) => piece.owner === player)
          .flatMap((piece) => getLegalMoves(piece.id, probe, field).flatMap((destination) => {
            const result = applyMove(piece.id, destination, probe, { analyzeCheckmate: false });
            return result.ok ? [{ piece, destination, after: result.state }] : [];
          }));
      }

      function completeRescueAction(state, player) {
        const hint = findClosestPlayableConfiguration(player, state);
        if (!hint) return null;
        const components = structuredClone(state.components);
        components[player] = structuredClone(hint.components);
        const activationOrders = structuredClone(state.activationOrders);
        activationOrders[player] = activationOrderForProfile(hint.components);
        const tunedBase = { ...state, components, activationOrders, selectedPieceId: null };
        const tuned = markInstability(tunedBase, evaluateField(tunedBase));
        const piece = tuned.pieces.find((candidate) => candidate.id === hint.pieceId);
        if (!piece) return null;
        const result = applyMove(hint.pieceId, hint.destination, tuned, { analyzeCheckmate: true });
        return result.ok ? { choice: { piece, destination: hint.destination, after: result.state }, score: 0, bot: "complete_rescue" } : null;
      }

      function graphScore(state, player) {
        const pieces = state.pieces.filter((piece) => piece.owner === player);
        let support = 0;
        let distance = 0;
        let pairs = 0;
        for (let i = 0; i < pieces.length; i += 1) {
          for (let j = i + 1; j < pieces.length; j += 1) {
            const d = Math.abs(pieces[i].position.x - pieces[j].position.x) + Math.abs(pieces[i].position.y - pieces[j].position.y);
            distance += d;
            pairs += 1;
            if (d <= 2) support += 1;
          }
        }
        return { support, averageDistance: pairs === 0 ? 0 : distance / pairs };
      }

      function fieldScore(state, player) {
        const field = evaluateField(state);
        let score = 0;
        for (const row of field) {
          for (const value of row) {
            if (value > 0) score += player === "red" ? 1 : -1;
            if (value < 0) score += player === "blue" ? 1 : -1;
          }
        }
        return score;
      }

      function abstractStateKey(state) {
        const blueKing = state.pieces.find((piece) => piece.owner === "blue" && piece.type === "king");
        const redKing = state.pieces.find((piece) => piece.owner === "red" && piece.type === "king");
        return [
          state.currentPlayer,
          blueKing ? blueKing.position.x + "," + blueKing.position.y : "x",
          redKing ? redKing.position.x + "," + redKing.position.y : "x",
          material(state, "blue"),
          material(state, "red"),
          rawKingMoves(state, "blue"),
          rawKingMoves(state, "red"),
          Math.floor(pressureAgainst(state, "blue") / 2),
          Math.floor(pressureAgainst(state, "red") / 2),
        ].join("|");
      }

      function recentAbstractCounts(state) {
        const counts = new Map();
        for (const entry of state.history.slice(-24)) {
          const probe = {
            ...state,
            pieces: structuredClone(entry.pieces),
            currentPlayer: entry.currentPlayer,
            components: structuredClone(entry.components),
            activationOrders: structuredClone(entry.activationOrders),
            status: entry.status,
            selectedPieceId: entry.selectedPieceId,
            turnNumber: entry.turnNumber,
            definitions: structuredClone(entry.definitions),
            waveScales: structuredClone(entry.waveScales),
            homeEnergy: structuredClone(entry.homeEnergy),
          };
          const key = abstractStateKey(probe);
          counts.set(key, (counts.get(key) ?? 0) + 1);
        }
        counts.set(abstractStateKey(state), (counts.get(abstractStateKey(state)) ?? 0) + 1);
        return counts;
      }

      function moveScore(bot, before, choice, player, random) {
        const enemy = opponent(player);
        const beforeOwnKing = rawKingMoves(before, player);
        const afterOwnKing = rawKingMoves(choice.after, player);
        const beforeEnemyKing = rawKingMoves(before, enemy);
        const afterEnemyKing = rawKingMoves(choice.after, enemy);
        const ownLost = Math.max(0, material(before, player) - material(choice.after, player));
        const enemyLost = Math.max(0, material(before, enemy) - material(choice.after, enemy));
        const materialScore = enemyLost * 14 - ownLost * 18;
        const ownSupportDelta = graphScore(choice.after, player).support - graphScore(before, player).support;
        const enemyPressureDelta = pressureAgainst(choice.after, enemy) - pressureAgainst(before, enemy);
        const fieldDelta = fieldScore(choice.after, player) - fieldScore(before, player);
        const ownDelta = afterOwnKing - beforeOwnKing;
        const enemyDelta = afterEnemyKing - beforeEnemyKing;
        const winBonus = choice.after.status === player + "-won" ? 1_000_000 : 0;
        const lossPenalty = choice.after.status === enemy + "-won" ? 1_000_000 : 0;
        if (bot === "random_ish") return random();
        if (bot === "easy") return winBonus - lossPenalty + random() * 4 + ownDelta * 10 - enemyDelta * 8 + materialScore * 0.25;
        if (bot === "medium") return winBonus - lossPenalty + ownDelta * 40 - enemyDelta * 70 + materialScore + fieldDelta * 5 + ownSupportDelta * 5;
        if (bot === "hard") return winBonus - lossPenalty + ownDelta * 70 - enemyDelta * 120 + materialScore * 1.2 + enemyPressureDelta * 14;
        if (bot === "converter") {
          const enemySafe = legalActions(choice.after, enemy).length;
          const ownSafe = legalActions(choice.after, player).length;
          const enemyField = evaluateField(withPlayer(choice.after, enemy));
          const enemyForced = isKingUnprotected(enemy, choice.after, enemyField)
            || getUnstablePieces(enemy, choice.after, enemyField).some((piece) => piece.type !== "king");
          const repeatPenalty = (recentAbstractCounts(before).get(abstractStateKey(choice.after)) ?? 0) * 900;
          const reopensTrapPenalty = beforeEnemyKing <= 2 && afterEnemyKing > beforeEnemyKing ? (afterEnemyKing - beforeEnemyKing) * 220 : 0;
          const trapPhase = beforeEnemyKing <= 3 || enemySafe <= 14;
          return winBonus - lossPenalty
            + (enemyForced ? 450 : 0)
            + Math.max(0, beforeEnemyKing - afterEnemyKing) * (trapPhase ? 180 : 65)
            - (trapPhase ? afterEnemyKing * 130 + enemySafe * 6 : enemyDelta * 42)
            + ownDelta * (trapPhase ? 18 : 28)
            + Math.max(0, ownSafe - enemySafe) * 3
            + materialScore * 1.1
            + fieldDelta * (trapPhase ? 1 : 7)
            - repeatPenalty
            - reopensTrapPenalty;
        }
        if (bot === "rescue_first") return winBonus - lossPenalty + ownDelta * 140 + ownSupportDelta * 10 - ownLost * 35;
        if (bot === "trap_first") return winBonus - lossPenalty - enemyDelta * 160 + enemyPressureDelta * 25 + materialScore;
        if (bot === "field_control") return winBonus - lossPenalty + fieldDelta * 20 + ownSupportDelta * 12 - ownLost * 8;
        if (bot === "material_first") return winBonus - lossPenalty + materialScore * 1.8 + ownDelta * 20 - enemyDelta * 20;
        if (bot === "chaos") return winBonus - lossPenalty + Math.abs(fieldDelta) * 12 + Math.abs(enemyPressureDelta) * 20 - ownLost * 3 + random();
        return winBonus - lossPenalty + ownDelta * 40 - enemyDelta * 40 + materialScore;
      }

      function chooseMove(state, game, halfTurn) {
        const player = state.currentPlayer;
        const bot = bots[(game + halfTurn + (player === "red" ? 3 : 0)) % bots.length];
        const choices = legalActions(state, player);
        if (choices.length === 0) return completeRescueAction(state, player);
        const random = rngFrom(seed * 1_000_003 + game * 9176 + halfTurn * 131 + (player === "red" ? 41 : 19));
        return choices
          .map((choice) => ({ choice, score: moveScore(bot, state, choice, player, random), bot }))
          .sort((left, right) => right.score - left.score)[0];
      }

      function earliest(current, value) {
        return current === null ? value : current;
      }

      function decisiveSignals(state, halfTurn) {
        const signals = [];
        for (const player of ["blue", "red"]) {
          const field = evaluateField(withPlayer(state, player));
          const kingMoves = rawKingMoves(state, player);
          const pressure = pressureAgainst(state, player);
          if (isKingUnprotected(player, state, field)) signals.push([player + "Check", halfTurn]);
          if (halfTurn > 0 && kingMoves === 0 && pressure > 0) signals.push([player + "ZeroEscapes", halfTurn]);
          if (kingMoves <= 1 && pressure > 0) signals.push([player + "NearCollapse", halfTurn]);
          if (getUnstablePieces(player, state, field).some((piece) => piece.type !== "king")) signals.push([player + "Unstable", halfTurn]);
        }
        return signals;
      }

      const results = [];
      const checkpointRows = new Map(checkpoints.map((checkpoint) => [checkpoint, { games: 0, winners: 0, nearCollapse: 0, zeroEscapes: 0, check: 0, materialLoss: 0 }]));
      for (let game = 0; game < games; game += 1) {
        let state = createInitialState();
        const row = {
          game,
          winner: null,
          stalledPlayer: null,
          stalledHalfTurn: null,
          terminalHalfTurn: null,
          firstCheck: null,
          firstZeroEscapes: null,
          firstNearCollapse: null,
          firstUnstable: null,
          firstMaterialLoss: null,
          finalHalfTurn: 0,
          finalBlueEscapes: 0,
          finalRedEscapes: 0,
          finalMessage: "",
        };
        let previousBlueMaterial = material(state, "blue");
        let previousRedMaterial = material(state, "red");

        for (let halfTurn = 0; state.status === "playing" && halfTurn < maxHalfTurns; halfTurn += 1) {
          for (const [kind, at] of decisiveSignals(state, halfTurn)) {
            if (kind.endsWith("Check")) row.firstCheck = earliest(row.firstCheck, at);
            if (kind.endsWith("ZeroEscapes")) row.firstZeroEscapes = earliest(row.firstZeroEscapes, at);
            if (kind.endsWith("NearCollapse")) row.firstNearCollapse = earliest(row.firstNearCollapse, at);
            if (kind.endsWith("Unstable")) row.firstUnstable = earliest(row.firstUnstable, at);
          }
          const selected = chooseMove(state, game, halfTurn);
          if (!selected) {
            const adjudicated = beginTurn(state);
            if (adjudicated.status !== "playing") {
              state = adjudicated;
              row.finalHalfTurn = halfTurn;
              break;
            }
            row.stalledPlayer = state.currentPlayer;
            row.stalledHalfTurn = halfTurn;
            break;
          }
          state = selected.choice.after;
          const blueMaterial = material(state, "blue");
          const redMaterial = material(state, "red");
          if ((blueMaterial < previousBlueMaterial || redMaterial < previousRedMaterial) && row.firstMaterialLoss === null) {
            row.firstMaterialLoss = halfTurn + 1;
          }
          previousBlueMaterial = blueMaterial;
          previousRedMaterial = redMaterial;
          row.finalHalfTurn = halfTurn + 1;
          for (const checkpoint of checkpoints) {
            if (halfTurn + 1 === checkpoint) {
              const bucket = checkpointRows.get(checkpoint);
              bucket.games += 1;
              if (state.status !== "playing") bucket.winners += 1;
              if (row.firstNearCollapse !== null) bucket.nearCollapse += 1;
              if (row.firstZeroEscapes !== null) bucket.zeroEscapes += 1;
              if (row.firstCheck !== null) bucket.check += 1;
              if (row.firstMaterialLoss !== null) bucket.materialLoss += 1;
            }
          }
        }
        if (state.status !== "playing") {
          row.winner = state.status === "blue-won" ? "blue" : "red";
          row.terminalHalfTurn = row.finalHalfTurn;
        }
        row.finalBlueEscapes = rawKingMoves(state, "blue");
        row.finalRedEscapes = rawKingMoves(state, "red");
        row.finalMessage = state.message;
        results.push(row);
        if ((game + 1) % 20 === 0 || game + 1 === games) console.error("probed " + (game + 1) + "/" + games);
      }

      function finite(values) {
        return values.filter((value) => value !== null && Number.isFinite(value)).sort((a, b) => a - b);
      }

      function quantiles(values) {
        const sorted = finite(values);
        if (sorted.length === 0) return null;
        const at = (p) => sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * p))];
        return { count: sorted.length, min: sorted[0], p25: at(0.25), median: at(0.5), p75: at(0.75), p90: at(0.9), max: sorted.at(-1) };
      }

      function average(values) {
        return values.length === 0 ? 0 : values.reduce((total, value) => total + value, 0) / values.length;
      }

        const winners = results.filter((row) => row.winner);
      const stalls = results.filter((row) => row.stalledHalfTurn !== null);
      const summary = {
        games,
        maxHalfTurns,
        seed,
        winners: {
          total: winners.length,
          blue: winners.filter((row) => row.winner === "blue").length,
          red: winners.filter((row) => row.winner === "red").length,
        },
        stalls: {
          total: stalls.length,
          blueToMove: stalls.filter((row) => row.stalledPlayer === "blue").length,
          redToMove: stalls.filter((row) => row.stalledPlayer === "red").length,
          halfTurns: quantiles(stalls.map((row) => row.stalledHalfTurn)),
        },
        finalHalfTurns: quantiles(results.map((row) => row.finalHalfTurn)),
        terminalHalfTurns: quantiles(results.map((row) => row.terminalHalfTurn)),
        firstCheck: quantiles(results.map((row) => row.firstCheck)),
        firstZeroEscapes: quantiles(results.map((row) => row.firstZeroEscapes)),
        firstNearCollapse: quantiles(results.map((row) => row.firstNearCollapse)),
        firstUnstable: quantiles(results.map((row) => row.firstUnstable)),
        firstMaterialLoss: quantiles(results.map((row) => row.firstMaterialLoss)),
        finalEscapes: {
          blueAverage: average(results.map((row) => row.finalBlueEscapes)),
          redAverage: average(results.map((row) => row.finalRedEscapes)),
          blueZero: results.filter((row) => row.finalBlueEscapes === 0).length,
          redZero: results.filter((row) => row.finalRedEscapes === 0).length,
        },
        checkpoints: [...checkpointRows.entries()].map(([halfTurn, row]) => ({ halfTurn, ...row })),
        finalMessages: Object.entries(results.reduce((counts, row) => {
          counts[row.finalMessage] = (counts[row.finalMessage] ?? 0) + 1;
          return counts;
        }, {})).sort((a, b) => b[1] - a[1]).slice(0, 12),
      };

      console.log(JSON.stringify(summary, null, 2));
    `,
    resolveDir: process.cwd(),
    sourcefile: "decisiveness-probe.ts",
    loader: "ts",
  },
  bundle: true,
  format: "esm",
  platform: "node",
  outfile,
  logLevel: "silent",
});

await import(pathToFileURL(outfile).href);
