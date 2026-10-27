import react from "@vitejs/plugin-react";
import { sites } from "@openai/sites-vite-plugin";
import { defineConfig } from "vite";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

const engineBuildId = createHash("sha256")
  .update(readFileSync(new URL("./public/engine/pkg/wave_field_engine.js", import.meta.url)))
  .update(readFileSync(new URL("./public/engine/pkg/wave_field_engine_bg.wasm", import.meta.url)))
  .digest("hex").slice(0, 16);

export default defineConfig({
  base: process.env.GITHUB_ACTIONS ? "/wave-field/" : "/",
  define: { __ENGINE_BUILD_ID__: JSON.stringify(engineBuildId) },
  plugins: [react(), sites()],
});
