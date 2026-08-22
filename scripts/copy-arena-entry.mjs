import { copyFile, mkdir } from "node:fs/promises";

await mkdir("dist/arena", { recursive: true });
await copyFile("dist/index.html", "dist/arena/index.html");
await mkdir("dist/hard", { recursive: true });
await copyFile("dist/index.html", "dist/hard/index.html");
await mkdir("dist/big", { recursive: true });
await copyFile("dist/index.html", "dist/big/index.html");
await mkdir("dist/optim-test", { recursive: true });
await copyFile("dist/index.html", "dist/optim-test/index.html");
await mkdir("dist/easy-test", { recursive: true });
await copyFile("dist/index.html", "dist/easy-test/index.html");
await mkdir("dist/low-rescue-test", { recursive: true });
await copyFile("dist/index.html", "dist/low-rescue-test/index.html");
