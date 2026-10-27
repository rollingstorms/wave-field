import react from "@vitejs/plugin-react";
import { sites } from "@openai/sites-vite-plugin";
import { defineConfig } from "vite";
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";

const engineFiles = ["wave_field_engine.js", "wave_field_engine_bg.wasm"]
  .map((file) => new URL(`./public/engine/pkg/${file}`, import.meta.url));
// Native tests load this config before the separate WASM build step.
const engineBuildId = engineFiles.every((file) => existsSync(file))
  ? engineFiles.reduce((hash, file) => hash.update(readFileSync(file)), createHash("sha256"))
    .digest("hex").slice(0, 16)
  : "unbuilt-engine";

export default defineConfig({
  base: process.env.GITHUB_ACTIONS ? "/wave-field/" : "/",
  define: { __ENGINE_BUILD_ID__: JSON.stringify(engineBuildId) },
  plugins: [react(), sites()],
});
