import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const budget = Number(process.env.AI_BUDGET_MS ?? 2500);
if (!Number.isFinite(budget) || budget < 50) throw new Error("AI_BUDGET_MS must be at least 50");
const build = spawnSync("cargo", ["build", "--release", "--manifest-path", "engine/Cargo.toml"], { cwd: root, stdio: "inherit" });
if (build.status !== 0) process.exit(build.status ?? 1);
const engines = [{ name: "current", path: resolve(root, "engine/target/release/wave-field-engine") }];
if (process.env.BASELINE_ENGINE) engines.unshift({ name: "baseline", path: resolve(process.env.BASELINE_ENGINE) });
for (const [name, file] of [
  ["opening", "product-initial-state.json"],
  ["three-control rescue", "sparse-coordinated-rescue.json"],
]) {
  const state = JSON.parse(readFileSync(resolve(root, "engine/tests", file), "utf8"));
  for (const engine of engines) {
    const result = spawnSync(engine.path, [], {
      cwd: root, encoding: "utf8", timeout: Math.max(15000, budget * 3),
      input: `${JSON.stringify({ method: "playTeacherTurns", policy: "hard", profile: true,
        turns: [{ state, player: state.currentPlayer, seed: 17, variety: 0, timeBudgetMs: budget }] })}\n`,
    });
    if (result.error || result.status !== 0) throw result.error ?? new Error(result.stderr);
    const { turns, profile } = JSON.parse(result.stdout);
    const next = turns[0].state;
    const controlsChanged = Object.keys(state.components.blue).reduce((total, kind) => total
      + state.components[state.currentPlayer][kind].filter((value, index) => value !== next.components[state.currentPlayer][kind][index]).length, 0);
    console.log(JSON.stringify({ case: name, engine: engine.name, budgetMs: budget,
      searchMs: Math.round(profile.searchMs), status: next.status, nextPlayer: next.currentPlayer,
      controlsChanged, rootProfiles: profile.rootTuningProfiles ?? null,
      completedDepth: profile.maxCompletedDepth, nodes: profile.nodes,
      candidates: profile.generatedCandidates }));
  }
}
