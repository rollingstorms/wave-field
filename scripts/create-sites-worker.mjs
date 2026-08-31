import { mkdir, writeFile } from "node:fs/promises";

const worker = `const htmlPaths = new Set([
  "/",
  "/arena",
  "/big",
  "/easy-test",
  "/hard",
  "/low-rescue-test",
  "/optim-test",
]);

function acceptsHtml(request) {
  return request.headers.get("accept")?.includes("text/html") ?? false;
}

function assetRequest(request, pathname) {
  const url = new URL(request.url);
  url.pathname = pathname;
  return new Request(url, request);
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const response = await env.ASSETS.fetch(request);
    if (response.status !== 404) return response;

    if (request.method !== "GET" || !acceptsHtml(request)) return response;

    if (htmlPaths.has(url.pathname)) {
      const path = url.pathname === "/" ? "/index.html" : \`\${url.pathname}/index.html\`;
      return env.ASSETS.fetch(assetRequest(request, path));
    }

    return env.ASSETS.fetch(assetRequest(request, "/index.html"));
  },
};
`;

await mkdir("dist/server", { recursive: true });
await writeFile("dist/server/index.js", worker);
