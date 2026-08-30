import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { build } from "esbuild";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const maxHalfTurns = Number(process.env.MAX_HALF_TURNS ?? 90);
const seed = Number(process.env.SEED ?? 123);
const timeBudgetMs = Number(process.env.AI_BUDGET_MS ?? 5);
const gamesPerPair = Number(process.env.GAMES_PER_PAIR ?? 1);
const out = resolve(process.env.OUT ?? "analysis/hard-param-sweep.jsonl");
const engineBin = process.env.ENGINE_BIN ?? "engine/target/release/wave-field-engine";
const outfile = "/tmp/wave-field-hard-param-sweep.mjs";

const requestedVariants = (process.env.VARIANTS ?? "")
  .split(",")
  .map((name) => name.trim())
  .filter(Boolean);

const allVariants = [
  { name: "baseline", conversion: 1.0, trap: 1.0, cycle: 1.0 },
  { name: "conservative", conversion: 0.7, trap: 0.8, cycle: 1.8 },
  { name: "high_trap", conversion: 1.35, trap: 1.6, cycle: 1.2 },
  { name: "max_trap", conversion: 1.8, trap: 2.2, cycle: 1.4 },
  { name: "anti_cycle", conversion: 1.0, trap: 1.0, cycle: 3.0 },
  { name: "low_conversion", conversion: 0.35, trap: 0.6, cycle: 0.6 },
];
const variants = requestedVariants.length
  ? allVariants.filter((variant) => requestedVariants.includes(variant.name))
  : allVariants;

await build({
  stdin: {
    contents: `
      import { spawn } from "node:child_process";
      import { createInterface } from "node:readline";
      import { appendFile, mkdir, writeFile } from "node:fs/promises";
      import { dirname } from "node:path";
      import { createInitialState } from "./src/game/initialState.ts";

      const maxHalfTurns = ${JSON.stringify(maxHalfTurns)};
      const seed = ${JSON.stringify(seed)};
      const timeBudgetMs = ${JSON.stringify(timeBudgetMs)};
      const gamesPerPair = ${JSON.stringify(gamesPerPair)};
      const out = ${JSON.stringify(out)};
      const engineBin = ${JSON.stringify(engineBin)};
      const variants = ${JSON.stringify(variants)};

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

      function winnerOf(state) {
        return state.status === "blue-won" ? "blue" : state.status === "red-won" ? "red" : null;
      }

      function terminalReason(state) {
        if (state.status === "playing") return "limit";
        if (state.message.includes("no legal rescue")) return "big_hat_no_rescue";
        if (state.message.includes("no legal move")) return "no_legal_move";
        if (state.message.includes("in check")) return "big_hat_check";
        return "terminal";
      }

      async function hardTurn(engine, state, player, variant, gameId, halfTurn) {
        return engine.request({
          method: "playHardTurn",
          state: { ...state, history: state.history.slice(-24) },
          player,
          seed: seed * 1000003 + gameId * 9176 + halfTurn * 131 + (player === "red" ? 41 : 19),
          variety: 0,
          timeBudgetMs,
          hardConversionWeight: variant.conversion,
          hardTrapFocus: variant.trap,
          hardCycleWeight: variant.cycle,
        });
      }

      async function playGame(engine, blue, red, gameId) {
        let state = createInitialState();
        const ply = [];
        for (let halfTurn = 0; halfTurn < maxHalfTurns && state.status === "playing"; halfTurn += 1) {
          const player = state.currentPlayer;
          const variant = player === "blue" ? blue : red;
          const beforeStatus = state.status;
          state = await hardTurn(engine, state, player, variant, gameId, halfTurn);
          ply.push({
            halfTurn,
            player,
            variant: variant.name,
            beforeStatus,
            afterStatus: state.status,
            message: state.message,
          });
        }
        const winner = winnerOf(state);
        return {
          game: gameId,
          blue: blue.name,
          red: red.name,
          blueParams: blue,
          redParams: red,
          winner,
          length: ply.length,
          terminalReason: terminalReason(state),
          losingSide: winner ? (winner === "blue" ? "red" : "blue") : null,
          lostWithinTwoAfterMove: winner && ply.length >= 2 ? ply.at(-2).variant : null,
        };
      }

      function pct(value, total) {
        return total ? (value / total * 100).toFixed(1) + "%" : "0.0%";
      }

      function median(values) {
        values = [...values].sort((a, b) => a - b);
        return values.length ? values[Math.floor(values.length / 2)] : 0;
      }

      const engine = new EngineClient();
      const rows = [];
      await mkdir(dirname(out), { recursive: true });
      await writeFile(out, "");
      try {
        let gameId = 0;
        for (const blue of variants) {
          for (const red of variants) {
            if (blue.name === red.name) continue;
            for (let repeat = 0; repeat < gamesPerPair; repeat += 1) {
              const row = await playGame(engine, blue, red, gameId++);
              rows.push(row);
              await appendFile(out, JSON.stringify(row) + "\\n");
              console.error("sweep " + rows.length + "/" + (variants.length * (variants.length - 1) * gamesPerPair) + " " + row.blue + " vs " + row.red + " -> " + (row.winner ?? "limit") + " in " + row.length);
            }
          }
        }
      } finally {
        engine.close();
      }

      const byVariant = new Map();
      function entry(name) {
        if (!byVariant.has(name)) byVariant.set(name, { games: 0, wins: 0, losses: 0, limits: 0, lengths: [], quickLosses: 0 });
        return byVariant.get(name);
      }
      for (const row of rows) {
        for (const [name, side] of [[row.blue, "blue"], [row.red, "red"]]) {
          const e = entry(name);
          e.games += 1;
          e.lengths.push(row.length);
          if (!row.winner) e.limits += 1;
          else if (row.winner === side) e.wins += 1;
          else e.losses += 1;
          if (row.lostWithinTwoAfterMove === name && row.winner !== side) e.quickLosses += 1;
        }
      }

      console.log("Hard hyperparameter sweep");
      console.log("variants=" + variants.length + " games=" + rows.length + " maxHalfTurns=" + maxHalfTurns + " budgetMs=" + timeBudgetMs);
      console.log("");
      console.log("Variant ranking");
      for (const [name, e] of [...byVariant.entries()].sort((a, b) => (b[1].wins / b[1].games) - (a[1].wins / a[1].games) || a[1].limits - b[1].limits)) {
        console.log("  " + name + ": games=" + e.games + " wins=" + e.wins + " winRate=" + pct(e.wins, e.games) + " losses=" + e.losses + " limits=" + e.limits + " medianLen=" + median(e.lengths) + " quickLossMarkers=" + e.quickLosses);
      }
      console.log("");
      console.log("Pair results");
      for (const row of rows) console.log("  " + row.blue + " vs " + row.red + ": " + (row.winner ?? "limit") + " len=" + row.length + " reason=" + row.terminalReason);
    `,
    resolveDir: process.cwd(),
    sourcefile: "wave-field-hard-param-sweep.ts",
    loader: "ts",
  },
  bundle: true,
  platform: "node",
  format: "esm",
  outfile,
  logLevel: "silent",
});

await import(pathToFileURL(outfile).href);
