import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const engineBinary = resolve(root, "engine/target/release/wave-field-engine");
const initialStatePath = resolve(root, "engine/tests/initial-state.json");

const options = {
  games: 500,
  heuristicGames: 50,
  maxPlies: 300,
  seed: 90210,
  variety: 0.55,
  timeBudgetMs: 10,
};

for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  const next = process.argv[index + 1];
  if (arg === "--games" && next) {
    options.games = Number(next);
    index += 1;
  } else if (arg === "--heuristic-games" && next) {
    options.heuristicGames = Number(next);
    index += 1;
  } else if (arg === "--max-plies" && next) {
    options.maxPlies = Number(next);
    index += 1;
  } else if (arg === "--seed" && next) {
    options.seed = Number(next);
    index += 1;
  } else if (arg === "--variety" && next) {
    options.variety = Number(next);
    index += 1;
  } else if (arg === "--time-budget-ms" && next) {
    options.timeBudgetMs = Number(next);
    index += 1;
  } else {
    console.error(`Unknown or incomplete option: ${arg}`);
    process.exit(1);
  }
}

function run(command, args, input) {
  const result = spawnSync(command, args, {
    cwd: root,
    encoding: "utf8",
    input,
    maxBuffer: 1024 * 1024 * 50,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    console.error(result.stderr);
    process.exit(result.status ?? 1);
  }
  return result.stdout;
}

function printSummary(label, payload, wallMs) {
  const batch = payload.batch ?? payload;
  const profile = payload.batch
    ? {
        candidateGenerationMs: payload.candidateGenerationMs,
        applyMoveMs: payload.applyMoveMs,
        avgCandidateGenerationMsPerPly: payload.avgCandidateGenerationMsPerPly,
        avgApplyMoveMsPerPly: payload.avgApplyMoveMsPerPly,
        avgCandidateMovesPerPly: payload.avgCandidateMovesPerPly,
      }
    : {};

  console.log(
    JSON.stringify(
      {
        label,
        wallMs,
        games: batch.games,
        elapsedMs: Number(batch.elapsedMs),
        totalPlies: batch.totalPlies,
        meanPlies: batch.meanPlies,
        decisive: batch.decisive,
        capped: batch.capped,
        redWins: batch.redWins,
        blueWins: batch.blueWins,
        msPerGame: batch.msPerGame,
        msPerPly: batch.msPerPly,
        pliesPerSecond: batch.pliesPerSecond,
        ...profile,
      },
      null,
      2,
    ),
  );
}

console.error("Building release Rust engine...");
run("cargo", ["build", "--manifest-path", "engine/Cargo.toml", "--release"]);

const state = JSON.parse(readFileSync(initialStatePath, "utf8"));
const runs = [
  {
    label: "random-rich",
    request: {
      method: "simulateRandomGames",
      state,
      games: options.games,
      maxPlies: options.maxPlies,
      seed: options.seed,
    },
  },
  {
    label: "random-lean",
    request: {
      method: "simulateRandomLeanGames",
      state,
      games: options.games,
      maxPlies: options.maxPlies,
      seed: options.seed,
    },
  },
  {
    label: "random-profile",
    request: {
      method: "profileRandomGames",
      state,
      games: options.games,
      maxPlies: options.maxPlies,
      seed: options.seed,
    },
  },
  {
    label: "heuristic",
    request: {
      method: "simulateAiGames",
      state,
      games: options.heuristicGames,
      maxPlies: options.maxPlies,
      seed: options.seed,
      variety: options.variety,
      timeBudgetMs: options.timeBudgetMs,
    },
  },
];

for (const { label, request } of runs) {
  const startedAt = Date.now();
  const output = run(engineBinary, [], `${JSON.stringify(request)}\n`);
  printSummary(label, JSON.parse(output), Date.now() - startedAt);
}
