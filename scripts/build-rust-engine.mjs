import { spawnSync } from "node:child_process";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const engine = resolve(root, "engine");
const output = resolve(root, "public/engine/pkg");
const wasm = resolve(engine, "target/wasm32-unknown-unknown/debug/wave_field_engine.wasm");

function run(command, args, cwd = root) {
  const result = spawnSync(command, args, { cwd, stdio: "inherit" });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}

mkdirSync(output, { recursive: true });
run("cargo", ["build", "--manifest-path", resolve(engine, "Cargo.toml"), "--target", "wasm32-unknown-unknown", "--features", "wasm"]);
run("wasm-bindgen", [wasm, "--target", "web", "--out-dir", output, "--out-name", "wave_field_engine"]);
