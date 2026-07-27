import { copyFile, mkdir } from "node:fs/promises";

await mkdir("dist/arena", { recursive: true });
await copyFile("dist/index.html", "dist/arena/index.html");
