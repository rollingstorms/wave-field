import react from "@vitejs/plugin-react";
import { sites } from "@openai/sites-vite-plugin";
import { defineConfig } from "vite";
import { execFileSync } from "node:child_process";

const engineBuildId = process.env.GITHUB_SHA
  ?? execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim();

export default defineConfig({
  base: process.env.GITHUB_ACTIONS ? "/wave-field/" : "/",
  define: { __ENGINE_BUILD_ID__: JSON.stringify(engineBuildId) },
  plugins: [react(), sites()],
});
