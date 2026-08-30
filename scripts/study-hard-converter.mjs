import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { build } from "esbuild";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const games = Number(process.env.GAMES ?? 120);
const maxHalfTurns = Number(process.env.MAX_HALF_TURNS ?? 240);
const seed = Number(process.env.SEED ?? 91);
const timeBudgetMs = Number(process.env.AI_BUDGET_MS ?? 250);
const out = resolve(process.env.OUT ?? "analysis/hard-converter-study.jsonl");
const engineBin = process.env.ENGINE_BIN ?? "engine/target/debug/wave-field-engine";
const fastMetrics = process.env.FULL_SAFE_METRICS !== "1";
const bots = (process.env.BOTS ?? "hard,hard_low_conversion,hard_high_trap,hard_cycle,random_ish,easy,medium,trap_first,rescue_first,field_control,material_first,chaos")
  .split(",")
  .map((bot) => bot.trim())
  .filter(Boolean);
const outfile = "/tmp/wave-field-hard-converter-study.mjs";

await build({
  stdin: {
    contents: `
      import { spawn } from "node:child_process";
      import { createInterface } from "node:readline";
      import { appendFile, mkdir, writeFile } from "node:fs/promises";
      import { dirname } from "node:path";
      import { evaluateField, evaluateSignedPieceContribution } from "./src/field/evaluateField.ts";
      import { createInitialState } from "./src/game/initialState.ts";
      import { getLegalMoves } from "./src/game/movement.ts";
      import { applyMove, opponent } from "./src/game/rules.ts";
      import { getUnstablePieces, isKingUnprotected } from "./src/game/victory.ts";

      const games = ${JSON.stringify(games)};
      const maxHalfTurns = ${JSON.stringify(maxHalfTurns)};
      const seed = ${JSON.stringify(seed)};
      const timeBudgetMs = ${JSON.stringify(timeBudgetMs)};
      const out = ${JSON.stringify(out)};
      const engineBin = ${JSON.stringify(engineBin)};
      const fastMetrics = ${JSON.stringify(fastMetrics)};
      const bots = ${JSON.stringify(bots)};
      const pieceTypes = ["pawn", "rook", "spy", "king"];
      const materialValue = { pawn: 2, rook: 4, spy: 3, king: 100 };

      class EngineClient {
        constructor() {
          this.child = spawn(engineBin, [], { stdio: ["pipe", "pipe", "inherit"] });
          this.pending = [];
          createInterface({ input: this.child.stdout }).on("line", (line) => {
            const next = this.pending.shift();
            if (!next) return;
            try {
              next.resolve(JSON.parse(line));
            } catch (error) {
              next.reject(error);
            }
          });
        }
        request(payload) {
          return new Promise((resolve, reject) => {
            this.pending.push({ resolve, reject });
            this.child.stdin.write(JSON.stringify(payload) + "\\n");
          });
        }
        close() {
          this.child.stdin.end();
          this.child.kill();
        }
      }

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

      function withPlayer(state, player) {
        return { ...state, currentPlayer: player };
      }

      function material(state, player) {
        return state.pieces
          .filter((piece) => piece.owner === player)
          .reduce((total, piece) => total + materialValue[piece.type], 0);
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
          return result.ok ? [{ ...action, after: result.state }] : [];
        });
      }

      function kingEscapes(state, player) {
        const king = state.pieces.find((piece) => piece.owner === player && piece.type === "king");
        if (!king) return 0;
        if (fastMetrics) return getLegalMoves(king.id, withPlayer(state, player), evaluateField(withPlayer(state, player))).length;
        return safeActions(state, player).filter((action) => action.piece.id === king.id).length;
      }

      function fieldControl(state) {
        const field = evaluateField(state);
        let red = 0;
        let blue = 0;
        let centerRed = 0;
        let centerBlue = 0;
        for (let y = 0; y < field.length; y += 1) {
          for (let x = 0; x < field[y].length; x += 1) {
            if (field[y][x] > 0) red += 1;
            if (field[y][x] < 0) blue += 1;
            if (x >= 2 && x <= 4 && y >= 2 && y <= 4) {
              if (field[y][x] > 0) centerRed += 1;
              if (field[y][x] < 0) centerBlue += 1;
            }
          }
        }
        return { red, blue, score: red - blue, centerScore: centerRed - centerBlue };
      }

      function pressureLinks(state, threshold = 1.5) {
        let redUnderPressure = 0;
        let blueUnderPressure = 0;
        const contributors = new Map();
        for (const target of state.pieces) {
          for (const source of state.pieces) {
            if (source.owner === target.owner) continue;
            const signed = evaluateSignedPieceContribution(source, target.position, state);
            const hostile = target.owner === "red" ? -signed : signed;
            if (hostile >= threshold) {
              if (target.owner === "red") redUnderPressure += 1;
              else blueUnderPressure += 1;
              const key = source.owner + ":" + source.type;
              contributors.set(key, (contributors.get(key) ?? 0) + 1);
            }
          }
        }
        return { redUnderPressure, blueUnderPressure, contributors: Object.fromEntries(contributors) };
      }

      function sideGraph(state, player) {
        const pieces = state.pieces.filter((piece) => piece.owner === player);
        const bins = new Map();
        let distanceSum = 0;
        let pairs = 0;
        let close2 = 0;
        let close3 = 0;
        for (let i = 0; i < pieces.length; i += 1) {
          for (let j = i + 1; j < pieces.length; j += 1) {
            const a = pieces[i];
            const b = pieces[j];
            const distance = Math.abs(a.position.x - b.position.x) + Math.abs(a.position.y - b.position.y);
            const types = [a.type, b.type].sort().join("-");
            const bucket = distance <= 1 ? "d1" : distance <= 2 ? "d2" : distance <= 3 ? "d3" : distance <= 5 ? "d4-5" : "d6+";
            bins.set(types + ":" + bucket, (bins.get(types + ":" + bucket) ?? 0) + 1);
            distanceSum += distance;
            pairs += 1;
            if (distance <= 2) close2 += 1;
            if (distance <= 3) close3 += 1;
          }
        }
        const king = pieces.find((piece) => piece.type === "king");
        const kingBins = new Map();
        if (king) {
          for (const piece of pieces) {
            if (piece.id === king.id) continue;
            const distance = Math.abs(king.position.x - piece.position.x) + Math.abs(king.position.y - piece.position.y);
            const bucket = distance <= 1 ? "d1" : distance <= 2 ? "d2" : distance <= 3 ? "d3" : distance <= 5 ? "d4-5" : "d6+";
            kingBins.set("king-" + piece.type + ":" + bucket, (kingBins.get("king-" + piece.type + ":" + bucket) ?? 0) + 1);
          }
        }
        return {
          pieces: pieces.length,
          averageDistance: pairs ? distanceSum / pairs : 0,
          close2,
          close3,
          connectedness: pieces.length <= 1 ? 1 : close2 / Math.max(1, pieces.length - 1),
          wl1: Object.fromEntries([...bins.entries()].sort()),
          kingRings: Object.fromEntries([...kingBins.entries()].sort()),
        };
      }

      function tuningKey(components) {
        return pieceTypes.map((pieceType) => {
          const values = components[pieceType].map((value) => value === 1 ? "+" : value === -1 ? "-" : "0").join("");
          return pieceType[0] + values;
        }).join("/");
      }

      function boardState(state) {
        return state.pieces
          .map((piece) => ({ id: piece.id, owner: piece.owner, type: piece.type, x: piece.position.x, y: piece.position.y, unstable: piece.unstable }))
          .sort((a, b) => a.id.localeCompare(b.id));
      }

      function movedPiece(before, after) {
        for (const piece of before.pieces) {
          const next = after.pieces.find((candidate) => candidate.id === piece.id);
          if (next && (piece.position.x !== next.position.x || piece.position.y !== next.position.y)) {
            return { id: piece.id, type: piece.type, from: piece.position, to: next.position };
          }
        }
        return null;
      }

      function tuneDistance(before, after, player) {
        let distance = 0;
        for (const pieceType of pieceTypes) {
          before.components[player][pieceType].forEach((value, index) => {
            if (value !== after.components[player][pieceType][index]) distance += 1;
          });
        }
        return distance;
      }

      function moveDelta(before, after, player) {
        const enemy = opponent(player);
        const beforeField = fieldControl(before);
        const afterField = fieldControl(after);
        return {
          ownEscapes: kingEscapes(after, player) - kingEscapes(before, player),
          enemyEscapes: kingEscapes(after, enemy) - kingEscapes(before, enemy),
          ownSafe: legalActions(after, player).length - legalActions(before, player).length,
          enemySafe: legalActions(after, enemy).length - legalActions(before, enemy).length,
          field: (player === "red" ? 1 : -1) * (afterField.score - beforeField.score),
          material: material(after, player) - material(before, player) - (material(after, enemy) - material(before, enemy)),
          ownSupport: sideGraph(after, player).close2 - sideGraph(before, player).close2,
          enemySupport: sideGraph(after, enemy).close2 - sideGraph(before, enemy).close2,
        };
      }

      function chooseCustom(bot, state, player, game, halfTurn) {
        const random = rngFrom(seed * 1000003 + game * 9176 + halfTurn * 131 + (player === "red" ? 41 : 19));
        const choices = safeActions(state, player);
        if (choices.length === 0) return null;
        if (bot === "random_ish") return choices[Math.floor(random() * choices.length)].after;
        const scored = choices.map((choice) => {
          const after = choice.after;
          const delta = moveDelta(state, after, player);
          const enemy = opponent(player);
          const win = after.status === player + "-won" ? 1_000_000 : 0;
          const loss = after.status === enemy + "-won" ? 1_000_000 : 0;
          let score = win - loss + random() * 0.01;
          if (bot === "easy") score += delta.ownEscapes * 10 - delta.enemyEscapes * 8 + delta.material * 4;
          else if (bot === "medium") score += delta.ownEscapes * 40 - delta.enemyEscapes * 70 + delta.material * 12 + delta.field * 5 + delta.ownSupport * 5;
          else if (bot === "trap_first") score += -delta.enemyEscapes * 150 - delta.enemySafe * 18 + delta.material * 8;
          else if (bot === "rescue_first") score += delta.ownEscapes * 140 + delta.ownSafe * 12 - Math.max(0, -delta.material) * 16;
          else if (bot === "field_control") score += delta.field * 20 + delta.ownSupport * 12 - delta.enemySupport * 8;
          else if (bot === "material_first") score += delta.material * 20 + delta.ownEscapes * 15 - delta.enemyEscapes * 15;
          else if (bot === "chaos") score += Math.abs(delta.field) * 16 + Math.abs(delta.enemyEscapes) * 25 - Math.max(0, -delta.material) * 4;
          else score += delta.ownEscapes * 35 - delta.enemyEscapes * 45 + delta.material * 10;
          return { after, score };
        });
        return scored.sort((a, b) => b.score - a.score)[0].after;
      }

      function hardParams(bot) {
        if (bot === "hard_low_conversion") return { hardConversionWeight: 0.35, hardTrapFocus: 0.6, hardCycleWeight: 0.6 };
        if (bot === "hard_high_trap") return { hardConversionWeight: 1.45, hardTrapFocus: 1.8, hardCycleWeight: 1.2 };
        if (bot === "hard_cycle") return { hardConversionWeight: 1.0, hardTrapFocus: 1.0, hardCycleWeight: 2.8 };
        return { hardConversionWeight: 1.0, hardTrapFocus: 1.0, hardCycleWeight: 1.0 };
      }

      function isHard(bot) {
        return bot.startsWith("hard");
      }

      async function chooseMove(engine, bot, state, player, game, halfTurn) {
        if (!isHard(bot)) return chooseCustom(bot, state, player, game, halfTurn);
        const engineState = { ...state, history: state.history.slice(-24) };
        return engine.request({
          method: "playHardTurn",
          state: engineState,
          player,
          seed: seed * 1000003 + game * 9176 + halfTurn * 131 + (player === "red" ? 41 : 19),
          variety: 0,
          timeBudgetMs,
          ...hardParams(bot),
        });
      }

      function terminalReason(state) {
        if (state.status === "playing") return "limit";
        return state.message.includes("no legal rescue") ? "big_hat_no_rescue"
          : state.message.includes("no legal move") ? "no_legal_move"
          : state.message.includes("in check") ? "big_hat_check"
          : "terminal";
      }

      function summarizeRows(rows) {
        const gamesById = new Map();
        for (const row of rows) {
          if (!gamesById.has(row.game)) gamesById.set(row.game, []);
          gamesById.get(row.game).push(row);
        }
        const summaries = [...gamesById.values()].map((game) => game.sort((a, b) => a.halfTurn - b.halfTurn).at(-1));
        const byBot = new Map();
        const byPair = new Map();
        const featureLift = new Map();
        const tuningLift = new Map();

        function inc(map, key, patch) {
          const entry = map.get(key) ?? { seen: 0, wins: 0, limits: 0 };
          entry.seen += patch.seen ?? 0;
          entry.wins += patch.wins ?? 0;
          entry.limits += patch.limits ?? 0;
          map.set(key, entry);
        }

        for (const final of summaries) {
          const winner = final.winner;
          for (const [bot, player] of [[final.blueBot, "blue"], [final.redBot, "red"]]) {
            inc(byBot, bot, { seen: 1, wins: winner === player ? 1 : 0, limits: winner ? 0 : 1 });
          }
          inc(byPair, final.blueBot + " vs " + final.redBot, { seen: 1, wins: winner === "blue" ? 1 : 0, limits: winner ? 0 : 1 });
        }

        for (const row of rows.filter((row) => row.winner)) {
          for (const side of ["blue", "red"]) {
            const sideWon = row.winner === side;
            const graph = row.graph[side];
            const tokens = [
              "avgD:" + Math.round(graph.averageDistance),
              "close2:" + Math.min(6, graph.close2),
              ...Object.entries(graph.kingRings).map(([key, value]) => key + "=" + value),
            ];
            for (const token of tokens) inc(featureLift, token, { seen: 1, wins: sideWon ? 1 : 0 });
            inc(tuningLift, side + ":" + row.tuning[side], { seen: 1, wins: sideWon ? 1 : 0 });
          }
        }

        const rate = (entry) => entry.seen ? entry.wins / entry.seen : 0;
        const top = (map, minSeen = 1) => [...map.entries()]
          .filter(([, entry]) => entry.seen >= minSeen)
          .sort((a, b) => rate(b[1]) - rate(a[1]) || b[1].seen - a[1].seen)
          .slice(0, 14);
        const bottom = (map, minSeen = 1) => [...map.entries()]
          .filter(([, entry]) => entry.seen >= minSeen)
          .sort((a, b) => rate(a[1]) - rate(b[1]) || b[1].seen - a[1].seen)
          .slice(0, 14);

        console.log("Hard converter study");
        console.log("games=" + summaries.length + " turns=" + rows.length + " maxHalfTurns=" + maxHalfTurns + " budgetMs=" + timeBudgetMs);
        console.log("");
        console.log("Bot game outcomes");
        for (const [bot, entry] of [...byBot.entries()].sort((a, b) => rate(b[1]) - rate(a[1]))) {
          console.log("  " + bot + ": games=" + entry.seen + " wins=" + entry.wins + " winRate=" + (rate(entry) * 100).toFixed(1) + "% limits=" + entry.limits);
        }
        console.log("");
        console.log("Pairings, blue win rate shown");
        for (const [pair, entry] of [...byPair.entries()].sort((a, b) => b[1].seen - a[1].seen || a[0].localeCompare(b[0])).slice(0, 24)) {
          console.log("  " + pair + ": n=" + entry.seen + " blueWins=" + entry.wins + " blueWinRate=" + (rate(entry) * 100).toFixed(1) + "% limits=" + entry.limits);
        }
        console.log("");
        console.log("Strong graph/WL-style tokens");
        for (const [key, entry] of top(featureLift, 80)) console.log("  " + key + ": seen=" + entry.seen + " winRate=" + (rate(entry) * 100).toFixed(1) + "%");
        console.log("");
        console.log("Weak graph/WL-style tokens");
        for (const [key, entry] of bottom(featureLift, 80)) console.log("  " + key + ": seen=" + entry.seen + " winRate=" + (rate(entry) * 100).toFixed(1) + "%");
        console.log("");
        console.log("Strong tuning patterns");
        for (const [key, entry] of top(tuningLift, 30)) console.log("  " + key + ": seen=" + entry.seen + " winRate=" + (rate(entry) * 100).toFixed(1) + "%");
        console.log("");
        console.log("Weak tuning patterns");
        for (const [key, entry] of bottom(tuningLift, 30)) console.log("  " + key + ": seen=" + entry.seen + " winRate=" + (rate(entry) * 100).toFixed(1) + "%");
      }

      const engine = new EngineClient();
      const rows = [];
      await mkdir(dirname(out), { recursive: true });
      await writeFile(out, "");
      try {
        for (let game = 0; game < games; game += 1) {
          const blueBot = bots[game % bots.length];
          const redBot = bots[Math.floor(game / bots.length) % bots.length];
          let state = createInitialState();
          const gameRows = [];
          for (let halfTurn = 0; halfTurn < maxHalfTurns && state.status === "playing"; halfTurn += 1) {
            const before = state;
            const player = before.currentPlayer;
            const bot = player === "blue" ? blueBot : redBot;
            const field = evaluateField(before);
            const next = await chooseMove(engine, bot, before, player, game, halfTurn);
            state = next ?? { ...before, status: opponent(player) + "-won", message: player + " has no legal move" };
            const moved = movedPiece(before, state);
            const row = {
              game,
              halfTurn,
              turn: before.turnNumber,
              player,
              bot,
              blueBot,
              redBot,
              move: moved ? moved.type + ":" + coord(moved.to) : "none",
              tuneDistance: tuneDistance(before, state, player),
              winner: null,
              terminalReason: null,
              legalMoves: legalActions(before, player).length,
              safeMoves: fastMetrics ? legalActions(before, player).length : safeActions(before, player).length,
              blueEscapes: kingEscapes(before, "blue"),
              redEscapes: kingEscapes(before, "red"),
              blueSafeMoves: fastMetrics ? legalActions(before, "blue").length : safeActions(before, "blue").length,
              redSafeMoves: fastMetrics ? legalActions(before, "red").length : safeActions(before, "red").length,
              unstable: {
                blue: getUnstablePieces("blue", before, field).length,
                red: getUnstablePieces("red", before, field).length,
              },
              bigHatInCheck: {
                blue: isKingUnprotected("blue", before, field),
                red: isKingUnprotected("red", before, field),
              },
              material: { blue: material(before, "blue"), red: material(before, "red") },
              fieldControl: fieldControl(before),
              pressure: pressureLinks(before),
              graph: { blue: sideGraph(before, "blue"), red: sideGraph(before, "red") },
              tuning: { blue: tuningKey(before.components.blue), red: tuningKey(before.components.red) },
              boardState: boardState(before),
            };
            gameRows.push(row);
          }
          const winner = state.status === "blue-won" ? "blue" : state.status === "red-won" ? "red" : null;
          for (const row of gameRows) {
            row.winner = winner;
            row.terminalReason = terminalReason(state);
            row.gameLength = gameRows.length;
            rows.push(row);
          }
          await appendFile(out, gameRows.map((row) => JSON.stringify(row)).join("\\n") + "\\n");
          console.error("studied " + (game + 1) + "/" + games + " games, rows=" + rows.length);
        }
      } finally {
        engine.close();
      }
      summarizeRows(rows);
    `,
    resolveDir: process.cwd(),
    sourcefile: "wave-field-hard-converter-study.ts",
    loader: "ts",
  },
  bundle: true,
  platform: "node",
  format: "esm",
  outfile,
  logLevel: "silent",
});

await import(pathToFileURL(outfile).href);
