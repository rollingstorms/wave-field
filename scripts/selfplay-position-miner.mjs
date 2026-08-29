import { build } from "esbuild";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const games = Number(process.env.GAMES ?? 200);
const maxHalfTurns = Number(process.env.MAX_HALF_TURNS ?? 140);
const seed = Number(process.env.SEED ?? 1);
const timeBudgetMs = Number(process.env.AI_BUDGET_MS ?? 90);
const analyzeCandidates = process.env.ANALYZE_CANDIDATES === "1";
const useBuiltinAi = process.env.USE_BUILTIN_AI === "1";
const fastMetrics = process.env.FAST_METRICS === "1";
const out = resolve(process.env.OUT ?? "analysis/selfplay-position-miner.jsonl");
const bots = (process.env.BOTS ?? "random_ish,easy,medium,hard,converter,rescue_first,trap_first,field_control,material_first,chaos")
  .split(",")
  .map((bot) => bot.trim())
  .filter(Boolean);
const botBlue = process.env.BOT_BLUE?.trim() || null;
const botRed = process.env.BOT_RED?.trim() || null;
const outfile = "/tmp/wave-field-selfplay-position-miner.mjs";

await build({
  stdin: {
    contents: `
      if (!globalThis.structuredClone) {
        globalThis.structuredClone = (value) => JSON.parse(JSON.stringify(value));
      }

      import { mkdir, writeFile } from "node:fs/promises";
      import { dirname } from "node:path";
      import { evaluateField, evaluateSignedPieceContribution } from "./src/field/evaluateField.ts";
      import { isSquareCompatible } from "./src/field/projection.ts";
      import { createInitialState, snapshot } from "./src/game/initialState.ts";
      import { getLegalMoves } from "./src/game/movement.ts";
      import { applyMove, beginTurn, findClosestPlayableConfiguration, opponent } from "./src/game/rules.ts";
      import { activationOrderForProfile } from "./src/game/tuning.ts";
      import { playEasyTurn, playHardTurn, playHeuristicTurn } from "./src/game/ai.ts";
      import { getUnstablePieces, isKingUnprotected, markInstability } from "./src/game/victory.ts";

      const games = ${JSON.stringify(games)};
      const maxHalfTurns = ${JSON.stringify(maxHalfTurns)};
      const seed = ${JSON.stringify(seed)};
      const timeBudgetMs = ${JSON.stringify(timeBudgetMs)};
      const analyzeCandidates = ${JSON.stringify(analyzeCandidates)};
      const useBuiltinAi = ${JSON.stringify(useBuiltinAi)};
      const fastMetrics = ${JSON.stringify(fastMetrics)};
      const out = ${JSON.stringify(out)};
      const bots = ${JSON.stringify(bots)};
      const botBlue = ${JSON.stringify(botBlue)};
      const botRed = ${JSON.stringify(botRed)};
      const pieceTypes = ["pawn", "rook", "spy", "king"];
      const materialValue = { pawn: 2, rook: 4, spy: 3, king: 100 };
      const labels = { pawn: "Round Hat", rook: "Tower", spy: "Triangle Hat", king: "Big Hat" };

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

      function coord(position) {
        return String.fromCharCode(65 + position.x) + (7 - position.y);
      }

      function pieceLabel(piece) {
        return labels[piece.type] + "@" + coord(piece.position);
      }

      function boardState(state) {
        return state.pieces
          .map((piece) => ({
            id: piece.id,
            owner: piece.owner,
            type: piece.type,
            x: piece.position.x,
            y: piece.position.y,
            unstable: piece.unstable,
          }))
          .sort((left, right) => left.id.localeCompare(right.id));
      }

      function material(state, player) {
        return state.pieces
          .filter((piece) => piece.owner === player)
          .reduce((total, piece) => total + materialValue[piece.type], 0);
      }

      function materialText(state) {
        return material(state, "blue") + "-" + material(state, "red");
      }

      function lostPieces(before, after, owner) {
        const remaining = new Set(after.pieces.map((piece) => piece.id));
        return before.pieces
          .filter((piece) => piece.owner === owner && !remaining.has(piece.id))
          .map((piece) => ({ id: piece.id, type: piece.type, value: materialValue[piece.type] }));
      }

      function tuneDiff(before, after, player) {
        const changes = [];
        for (const pieceType of pieceTypes) {
          before.components[player][pieceType].forEach((value, index) => {
            const next = after.components[player][pieceType][index];
            if (value !== next) changes.push({ pieceType, componentIndex: index, from: value, to: next });
          });
        }
        return changes;
      }

      function withPlayer(state, player) {
        return { ...state, currentPlayer: player };
      }

      function legalActions(state, player = state.currentPlayer) {
        const probe = withPlayer(state, player);
        const field = evaluateField(probe);
        return probe.pieces
          .filter((piece) => piece.owner === player)
          .flatMap((piece) => getLegalMoves(piece.id, probe, field).map((destination) => ({ piece, destination })));
      }

      function safeActions(state, player = state.currentPlayer) {
        return legalActions(state, player).flatMap((action) => {
          const result = applyMove(action.piece.id, action.destination, withPlayer(state, player), { analyzeCheckmate: false });
          return result.ok ? [{ ...action, state: result.state }] : [];
        });
      }

      function kingEscapes(state, player) {
        const king = state.pieces.find((piece) => piece.owner === player && piece.type === "king");
        if (!king) return 0;
        const probe = withPlayer(state, player);
        return safeActions(probe, player).filter((action) => action.piece.id === king.id).length;
      }

      function rawKingMoves(state, player) {
        const king = state.pieces.find((piece) => piece.owner === player && piece.type === "king");
        if (!king) return 0;
        return getLegalMoves(king.id, withPlayer(state, player), evaluateField(withPlayer(state, player))).length;
      }

      function fieldControl(state, field = evaluateField(state)) {
        let redCells = 0;
        let blueCells = 0;
        let neutralCells = 0;
        let redNearBlueKing = 0;
        let blueNearRedKing = 0;
        let l1 = 0;
        const blueKing = state.pieces.find((piece) => piece.owner === "blue" && piece.type === "king");
        const redKing = state.pieces.find((piece) => piece.owner === "red" && piece.type === "king");
        for (let y = 0; y < field.length; y += 1) {
          for (let x = 0; x < field[y].length; x += 1) {
            const value = field[y][x];
            l1 += Math.abs(value);
            if (value > 0) redCells += 1;
            else if (value < 0) blueCells += 1;
            else neutralCells += 1;
            if (blueKing && Math.max(Math.abs(x - blueKing.position.x), Math.abs(y - blueKing.position.y)) <= 2) {
              redNearBlueKing += Math.max(0, value);
            }
            if (redKing && Math.max(Math.abs(x - redKing.position.x), Math.abs(y - redKing.position.y)) <= 2) {
              blueNearRedKing += Math.max(0, -value);
            }
          }
        }
        return {
          redCells,
          blueCells,
          neutralCells,
          score: redCells - blueCells,
          l1,
          redNearBlueKing,
          blueNearRedKing,
        };
      }

      function graphMetrics(state, player) {
        const pieces = state.pieces.filter((piece) => piece.owner === player);
        let pairDistance = 0;
        let pairs = 0;
        let closeLinks = 0;
        for (let i = 0; i < pieces.length; i += 1) {
          for (let j = i + 1; j < pieces.length; j += 1) {
            const distance = Math.abs(pieces[i].position.x - pieces[j].position.x) + Math.abs(pieces[i].position.y - pieces[j].position.y);
            pairDistance += distance;
            pairs += 1;
            if (distance <= 2) closeLinks += 1;
          }
        }
        return {
          averageDistance: pairs === 0 ? 0 : pairDistance / pairs,
          supportScore: closeLinks,
          connectedness: pieces.length <= 1 ? 1 : closeLinks / Math.max(1, pieces.length - 1),
        };
      }

      function pressureLinks(state, threshold = 1.5) {
        const links = [];
        for (const target of state.pieces) {
          for (const contributor of state.pieces) {
            if (contributor.owner === target.owner) continue;
            const signed = evaluateSignedPieceContribution(contributor, target.position, state);
            const hostile = target.owner === "red" ? -signed : signed;
            if (hostile >= threshold) {
              links.push({
                targetPieceID: target.id,
                targetOwner: target.owner,
                targetKind: target.type,
                contributorPieceID: contributor.id,
                contributorKind: contributor.type,
                magnitude: hostile,
              });
            }
          }
        }
        return links.sort((left, right) => right.magnitude - left.magnitude);
      }

      function sideMetrics(state, player) {
        const field = evaluateField(withPlayer(state, player));
        const graph = graphMetrics(state, player);
        const legalMoves = legalActions(state, player).length;
        return {
          legalMoves,
          safeMoves: fastMetrics ? legalMoves : safeActions(state, player).length,
          bigHatEscapes: fastMetrics ? rawKingMoves(state, player) : kingEscapes(state, player),
          unstablePieces: getUnstablePieces(player, state, field).length,
          material: material(state, player),
          supportScore: graph.supportScore,
          connectedness: graph.connectedness,
          averageAlliedDistance: graph.averageDistance,
          bigHatInCheck: isKingUnprotected(player, state, field),
        };
      }

      function featureSnapshot(state) {
        const field = evaluateField(state);
        const marked = markInstability(state, field);
        return {
          turnNumber: state.turnNumber,
          currentPlayer: state.currentPlayer,
          status: state.status,
          blue: sideMetrics(marked, "blue"),
          red: sideMetrics(marked, "red"),
          fieldControl: fieldControl(marked, field),
          pressureLinks: pressureLinks(marked).slice(0, 12),
        };
      }

      function componentDistance(left, right) {
        let distance = 0;
        for (const pieceType of pieceTypes) {
          left[pieceType].forEach((value, index) => {
            if (value !== right[pieceType][index]) distance += 1;
          });
        }
        return distance;
      }

      function profileAfterControlChange(state, player, pieceType, componentIndex, value) {
        const profile = [...state.components[player][pieceType]];
        const currentValue = profile[componentIndex];
        if (currentValue === value) return null;
        const activeIndices = profile.flatMap((coefficient, index) => coefficient === 0 ? [] : [index]);
        const nextOrder = state.activationOrders[player][pieceType]
          .filter((index) => activeIndices.includes(index) && index !== componentIndex);
        for (const index of activeIndices) {
          if (index !== componentIndex && !nextOrder.includes(index)) nextOrder.push(index);
        }
        const strength = pieceType === "pawn" || pieceType === "spy" ? 1 : 2;
        if (currentValue === 0 && activeIndices.length >= strength) {
          const evictedIndex = nextOrder.shift();
          if (evictedIndex !== undefined) profile[evictedIndex] = 0;
        }
        profile[componentIndex] = value;
        return profile;
      }

      function tuningCandidates(state, player, maxCandidates) {
        const current = state.components[player];
        const candidates = [structuredClone(current)];
        const seen = new Set([JSON.stringify(current)]);
        for (const pieceType of pieceTypes) {
          for (let index = 0; index < current[pieceType].length; index += 1) {
            for (const value of [-1, 1]) {
              const profile = profileAfterControlChange(state, player, pieceType, index, value);
              if (!profile) continue;
              const next = structuredClone(current);
              next[pieceType] = profile;
              const key = JSON.stringify(next);
              if (!seen.has(key)) {
                seen.add(key);
                candidates.push(next);
              }
              if (candidates.length >= maxCandidates) return candidates;
            }
          }
        }
        return candidates;
      }

      function tunedState(state, player, components) {
        const nextComponents = structuredClone(state.components);
        nextComponents[player] = structuredClone(components);
        const activationOrders = structuredClone(state.activationOrders);
        activationOrders[player] = activationOrderForProfile(components);
        return markInstability({ ...state, components: nextComponents, activationOrders, selectedPieceId: null }, evaluateField({ ...state, components: nextComponents }));
      }

      function candidateActions(state, player, maxTunings) {
        const choices = [];
        for (const components of tuningCandidates(state, player, maxTunings)) {
          const tuned = tunedState(state, player, components);
          const field = evaluateField(tuned);
          for (const piece of tuned.pieces.filter((candidate) => candidate.owner === player)) {
            for (const destination of getLegalMoves(piece.id, tuned, field)) {
              const result = applyMove(piece.id, destination, tuned, { analyzeCheckmate: false });
              if (result.ok) choices.push({ tuned, piece, destination, after: result.state });
            }
          }
        }
        return choices;
      }

      function currentTuningActions(state, player) {
        const field = evaluateField(state);
        return state.pieces
          .filter((piece) => piece.owner === player)
          .flatMap((piece) => getLegalMoves(piece.id, state, field).flatMap((destination) => {
            const result = applyMove(piece.id, destination, state, { analyzeCheckmate: false });
            return result.ok ? [{ tuned: state, piece, destination, after: result.state }] : [];
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
        return result.ok ? { tuned, piece, destination: hint.destination, after: result.state } : null;
      }

      function moveDelta(before, after, player) {
        const enemy = opponent(player);
        const beforeFieldControl = fieldControl(before);
        const afterFieldControl = fieldControl(after);
        const beforeOwnGraph = graphMetrics(before, player);
        const afterOwnGraph = graphMetrics(after, player);
        const beforeEnemyGraph = graphMetrics(before, enemy);
        const afterEnemyGraph = graphMetrics(after, enemy);
        const beforeOwnUnstable = getUnstablePieces(player, before, evaluateField(before)).length;
        const afterOwnUnstable = getUnstablePieces(player, after, evaluateField(after)).length;
        const beforeEnemyUnstable = getUnstablePieces(enemy, before, evaluateField(before)).length;
        const afterEnemyUnstable = getUnstablePieces(enemy, after, evaluateField(after)).length;
        const beforeField = evaluateField(before);
        const afterField = evaluateField(after);
        const ownLost = lostPieces(before, after, player).reduce((total, piece) => total + piece.value, 0);
        const enemyLost = lostPieces(before, after, enemy).reduce((total, piece) => total + piece.value, 0);
        return {
          ownBigHatEscapeDelta: rawKingMoves(after, player) - rawKingMoves(before, player),
          enemyBigHatEscapeDelta: rawKingMoves(after, enemy) - rawKingMoves(before, enemy),
          ownSafeMoveDelta: legalActions(after, player).length - legalActions(before, player).length,
          enemySafeMoveDelta: legalActions(after, enemy).length - legalActions(before, enemy).length,
          ownLost,
          enemyLost,
          fieldControlDelta: (player === "red" ? 1 : -1) * (afterFieldControl.score - beforeFieldControl.score),
          fieldL1Delta: afterFieldControl.l1 - beforeFieldControl.l1,
          pressureLinkDelta: 0,
          ownSupportDelta: afterOwnGraph.supportScore - beforeOwnGraph.supportScore,
          enemySupportDelta: afterEnemyGraph.supportScore - beforeEnemyGraph.supportScore,
          unstableDelta: (afterEnemyUnstable - beforeEnemyUnstable) - (afterOwnUnstable - beforeOwnUnstable),
          signSwing: Math.abs(afterFieldControl.score - beforeFieldControl.score),
          fieldMassSwing: Math.abs(afterField.flat().reduce((total, value) => total + Math.abs(value), 0)
            - beforeField.flat().reduce((total, value) => total + Math.abs(value), 0)),
        };
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
          Math.floor(legalActions(state, "blue").length / 4),
          Math.floor(legalActions(state, "red").length / 4),
          Math.floor(fieldControl(state).score / 6),
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

      function converterScore(choice, before, player, random) {
        const enemy = opponent(player);
        const beforeEnemyEscapes = rawKingMoves(before, enemy);
        const afterEnemyEscapes = rawKingMoves(choice.after, enemy);
        const beforeOwnEscapes = rawKingMoves(before, player);
        const afterOwnEscapes = rawKingMoves(choice.after, player);
        const enemySafe = legalActions(choice.after, enemy).length;
        const ownSafe = legalActions(choice.after, player).length;
        const enemyField = evaluateField(withPlayer(choice.after, enemy));
        const enemyForcedToRescue = isKingUnprotected(enemy, choice.after, enemyField)
          || getUnstablePieces(enemy, choice.after, enemyField).some((piece) => piece.type !== "king");
        const repeatPenalty = (recentAbstractCounts(before).get(abstractStateKey(choice.after)) ?? 0) * 900;
        const reopensTrapPenalty = beforeEnemyEscapes <= 2 && afterEnemyEscapes > beforeEnemyEscapes ? (afterEnemyEscapes - beforeEnemyEscapes) * 220 : 0;
        const closeExitBonus = Math.max(0, beforeEnemyEscapes - afterEnemyEscapes) * (beforeEnemyEscapes <= 3 ? 180 : 65);
        const trapPhase = beforeEnemyEscapes <= 3 || enemySafe <= 14;
        const materialSwing = lostPieces(before, choice.after, enemy).reduce((total, piece) => total + piece.value, 0)
          - lostPieces(before, choice.after, player).reduce((total, piece) => total + piece.value, 0);
        const winBonus = choice.after.status === player + "-won" ? 1_000_000 : 0;
        const lossPenalty = choice.after.status === enemy + "-won" ? 1_000_000 : 0;
        if (trapPhase) {
          return winBonus - lossPenalty
            + (enemyForcedToRescue ? 450 : 0)
            + closeExitBonus
            - afterEnemyEscapes * 130
            - enemySafe * 6
            + Math.max(0, afterOwnEscapes) * 18
            + Math.max(0, ownSafe - enemySafe) * 3
            + materialSwing * 18
            - reopensTrapPenalty
            - repeatPenalty
            + random();
        }
        const delta = moveDelta(before, choice.after, player);
        return winBonus - lossPenalty
          + closeExitBonus
          + delta.ownBigHatEscapeDelta * 28
          - delta.enemyBigHatEscapeDelta * 42
          + delta.fieldControlDelta * 7
          + delta.ownSupportDelta * 8
          + materialSwing * 14
          - repeatPenalty * 0.35
          - reopensTrapPenalty
          + random();
      }

      function scoreChoice(bot, choice, before, player, random) {
        const delta = moveDelta(before, choice.after, player);
        const tuneDistance = componentDistance(before.components[player], choice.tuned.components[player]);
        const noise = random() * 0.001;
        const winBonus = choice.after.status === player + "-won" ? 1_000_000 : 0;
        const survivalPenalty = choice.after.status === opponent(player) + "-won" ? 1_000_000 : 0;
        const materialScore = delta.enemyLost * 14 - delta.ownLost * 18;
        if (bot === "random_ish") return random();
        if (bot === "easy") return winBonus - survivalPenalty + random() * 4 + delta.ownBigHatEscapeDelta * 12 - delta.enemyBigHatEscapeDelta * 10 + materialScore * 0.35 - delta.fieldControlDelta * 0.25 + noise;
        if (bot === "medium") return winBonus - survivalPenalty - delta.enemyBigHatEscapeDelta * 80 + delta.ownBigHatEscapeDelta * 50 + materialScore + delta.fieldControlDelta * 6 + delta.ownSupportDelta * 5 + noise;
        if (bot === "hard") return winBonus - survivalPenalty - delta.enemyBigHatEscapeDelta * 130 + delta.ownBigHatEscapeDelta * 80 - delta.enemySafeMoveDelta * 10 + delta.ownSafeMoveDelta * 7 + materialScore * 1.2 + delta.unstableDelta * 18 + noise;
        if (bot === "converter") return converterScore(choice, before, player, random);
        if (bot === "rescue_first") return winBonus - survivalPenalty + delta.ownBigHatEscapeDelta * 120 + delta.ownSafeMoveDelta * 16 - delta.ownLost * 35 - tuneDistance * 2 + noise;
        if (bot === "trap_first") return winBonus - survivalPenalty - delta.enemyBigHatEscapeDelta * 150 - delta.enemySafeMoveDelta * 18 + delta.unstableDelta * 30 + materialScore + noise;
        if (bot === "field_control") return winBonus - survivalPenalty + delta.fieldControlDelta * 18 + delta.ownSupportDelta * 12 - delta.enemySupportDelta * 8 - delta.ownLost * 10 + noise;
        if (bot === "material_first") return winBonus - survivalPenalty + materialScore + delta.ownBigHatEscapeDelta * 20 - delta.enemyBigHatEscapeDelta * 20 + noise;
        if (bot === "chaos") return winBonus - survivalPenalty + delta.signSwing * 12 + delta.fieldMassSwing * 0.5 + Math.abs(delta.pressureLinkDelta) * 20 - delta.ownLost * 4 + noise;
        return winBonus - survivalPenalty - delta.enemyBigHatEscapeDelta * 80 + delta.ownBigHatEscapeDelta * 50 + materialScore + delta.fieldControlDelta * 6 + noise;
      }

      function chooseCustom(bot, state, player, game, halfTurn) {
        const random = rngFrom(seed * 1_000_003 + game * 9_176 + halfTurn * 131 + (player === "red" ? 41 : 19));
        const choices = currentTuningActions(state, player);
        if (choices.length === 0) return completeRescueAction(state, player);
        if (bot === "random_ish") return choices[Math.floor(random() * choices.length)];
        return choices
          .map((choice) => ({ choice, score: scoreChoice(bot, choice, state, player, random) }))
          .sort((left, right) => right.score - left.score)[0].choice;
      }

      function movedPiece(before, after) {
        for (const piece of before.pieces) {
          const next = after.pieces.find((candidate) => candidate.id === piece.id);
          if (next && (piece.position.x !== next.position.x || piece.position.y !== next.position.y)) {
            return { piece, from: piece.position, to: next.position };
          }
        }
        return null;
      }

      function playedByBuiltIn(bot, state, player, game, halfTurn) {
        const options = {
          seed: seed * 1_000_003 + game * 9_176 + halfTurn * 131 + (player === "red" ? 41 : 19),
          variety: bot === "easy" ? 0.45 : bot === "medium" ? 0.3 : 0.12,
          timeBudgetMs: bot === "hard" ? Math.max(250, timeBudgetMs * 4) : timeBudgetMs,
        };
        if (bot === "easy") return playEasyTurn(state, player, options);
        if (bot === "hard") return playHardTurn(state, player, options);
        return playHeuristicTurn(state, player, options);
      }

      function chooseMove(bot, state, player, game, halfTurn) {
        if (useBuiltinAi && (bot === "easy" || bot === "medium" || bot === "hard")) {
          const after = playedByBuiltIn(bot, state, player, game, halfTurn);
          const moved = movedPiece(state, after);
          const tuned = { ...state, components: after.components, activationOrders: after.activationOrders };
          return moved ? { tuned, piece: moved.piece, destination: moved.to, after } : null;
        }
        return chooseCustom(bot, state, player, game, halfTurn);
      }

      function candidateReason(player, bot, before, choice) {
        const enemy = opponent(player);
        const delta = moveDelta(before, choice.after, player);
        const parts = [];
        if (delta.enemyBigHatEscapeDelta < 0) parts.push("cuts " + enemy + " Big Hat escapes by " + Math.abs(delta.enemyBigHatEscapeDelta));
        if (delta.ownBigHatEscapeDelta > 0) parts.push("adds " + delta.ownBigHatEscapeDelta + " own Big Hat escapes");
        if (delta.enemyLost > 0) parts.push("wins " + delta.enemyLost + " material");
        if (delta.ownLost > 0) parts.push("spends " + delta.ownLost + " material");
        if (delta.fieldControlDelta > 0) parts.push("gains field control");
        if (delta.ownSupportDelta > 0) parts.push("improves support");
        if (parts.length === 0) parts.push("best " + bot.replaceAll("_", "-") + " heuristic score");
        return parts.slice(0, 3).join(", ");
      }

      function topBottomCandidates(state, player, bot, game, halfTurn) {
        if (!analyzeCandidates) return { top: [], bottom: [] };
        if (bot === "easy" || bot === "medium" || bot === "hard") return { top: [], bottom: [] };
        const random = rngFrom(seed * 17 + game * 97 + halfTurn * 13);
        const rows = currentTuningActions(state, player)
          .map((choice) => ({
            move: choice.piece.type + ":" + coord(choice.destination),
            pieceId: choice.piece.id,
            score: scoreChoice(bot, choice, state, player, random),
            delta: moveDelta(state, choice.after, player),
          }))
          .sort((left, right) => right.score - left.score);
        return {
          top: rows.slice(0, 3),
          bottom: rows.slice(-3).reverse(),
        };
      }

      function botFor(game, player) {
        if (player === "blue" && botBlue) return botBlue;
        if (player === "red" && botRed) return botRed;
        const offset = player === "blue" ? 0 : Math.floor(bots.length / 2) + 1;
        return bots[(game + offset) % bots.length];
      }

      const lines = [];
      for (let game = 0; game < games; game += 1) {
        let state = createInitialState();
        const records = [];
        for (let halfTurn = 0; state.status === "playing" && halfTurn < maxHalfTurns; halfTurn += 1) {
          const before = state;
          const player = before.currentPlayer;
          const bot = botFor(game, player);
          const beforeFeatures = featureSnapshot(before);
          const hint = beforeFeatures[player].bigHatInCheck
            ? findClosestPlayableConfiguration(player, markInstability(before, evaluateField(before)))
            : null;
          const choice = chooseMove(bot, before, player, game, halfTurn);
          if (!choice) {
            state = beginTurn(before);
            break;
          }
          state = choice.after;
          const move = {
            pieceId: choice.piece.id,
            pieceType: choice.piece.type,
            from: choice.piece.position,
            to: choice.destination,
            notation: choice.piece.type + ":" + coord(choice.destination),
          };
          const delta = moveDelta(before, state, player);
          const analysis = topBottomCandidates(before, player, bot, game, halfTurn);
          records.push({
            game,
            turn: before.turnNumber,
            halfTurn,
            player,
            bot,
            move,
            tune: tuneDiff(before, choice.tuned, player),
            boardState: boardState(before),
            components: before.components,
            legalMovesCount: beforeFeatures[player].legalMoves,
            safeMovesCount: beforeFeatures[player].safeMoves,
            blueEscapes: beforeFeatures.blue.bigHatEscapes,
            redEscapes: beforeFeatures.red.bigHatEscapes,
            blueSafeMoves: beforeFeatures.blue.safeMoves,
            redSafeMoves: beforeFeatures.red.safeMoves,
            piecesLost: {
              blue: lostPieces(before, state, "blue"),
              red: lostPieces(before, state, "red"),
            },
            unstablePieces: {
              blue: beforeFeatures.blue.unstablePieces,
              red: beforeFeatures.red.unstablePieces,
            },
            fieldControlScore: beforeFeatures.fieldControl.score,
            fieldControl: beforeFeatures.fieldControl,
            connectednessSupportScore: {
              blue: beforeFeatures.blue.supportScore,
              red: beforeFeatures.red.supportScore,
            },
            alliedAverageDistance: {
              blue: beforeFeatures.blue.averageAlliedDistance,
              red: beforeFeatures.red.averageAlliedDistance,
            },
            pressureLinksAboveThreshold: beforeFeatures.pressureLinks,
            hintSearchFoundRescue: Boolean(hint),
            hintSearchRescue: hint ? {
              pieceId: hint.pieceId,
              pieceType: hint.pieceType,
              destination: hint.destination,
              changedComponents: hint.changedComponents,
            } : null,
            bigHatEscapeDelta: {
              enemyBefore: beforeFeatures[opponent(player)].bigHatEscapes,
              enemyAfter: featureSnapshot(state)[opponent(player)].bigHatEscapes,
              enemyDelta: delta.enemyBigHatEscapeDelta,
              mineBefore: beforeFeatures[player].bigHatEscapes,
              mineAfter: featureSnapshot(state)[player].bigHatEscapes,
              mineDelta: delta.ownBigHatEscapeDelta,
            },
            moveAnalysis: {
              reason: labels[choice.piece.type] + " -> " + coord(choice.destination) + ": " + candidateReason(player, bot, before, choice),
              top: analysis.top,
              bottom: analysis.bottom,
            },
            material: materialText(before),
            winner: null,
            turnsUntilWin: null,
            terminalMessage: null,
          });
        }
        const winner = state.status === "red-won" ? "red" : state.status === "blue-won" ? "blue" : null;
        for (const record of records) {
          record.winner = winner;
          record.turnsUntilWin = winner ? records.length - record.halfTurn : null;
          record.terminalMessage = state.message;
          lines.push(JSON.stringify(record));
        }
        if ((game + 1) % 25 === 0 || game + 1 === games) {
          console.error("mined " + (game + 1) + "/" + games + " games, records=" + lines.length);
        }
      }
      await mkdir(dirname(out), { recursive: true });
      await writeFile(out, lines.join("\\n") + (lines.length > 0 ? "\\n" : ""));
      console.log("Wrote " + lines.length + " turns to " + out);
    `,
    resolveDir: process.cwd(),
    sourcefile: "selfplay-position-miner.ts",
    loader: "ts",
  },
  bundle: true,
  format: "esm",
  platform: "node",
  outfile,
  logLevel: "silent",
});

await import(pathToFileURL(outfile).href);
