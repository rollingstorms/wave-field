import type { Plugin } from 'vite';
import { readFile, readdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const directory = fileURLToPath(new URL('../training/tracking/selfplay-live/', import.meta.url));
export function selfplayViewerApi(): Plugin {
  return {
    name: 'local-selfplay-viewer',
    configureServer(server) {
      server.middlewares.use('/__selfplay', async (req, res) => {
        res.setHeader('Content-Type', 'application/json');
        res.setHeader('Cache-Control', 'no-store');
        if (req.method !== 'GET') { res.statusCode = 405; res.end('{}'); return; }
        try {
          const url = new URL(req.url ?? '/', 'http://localhost');
          if (url.pathname === '/games') {
            const files = await readdir(directory).catch(() => [] as string[]);
            const games = await Promise.all(files.filter(x => /^game-\d+\.meta\.json$/.test(x))
              .map(async x => JSON.parse(await readFile(`${directory}/${x}`, 'utf8'))));
            res.end(JSON.stringify({games: games.sort((a, b) => a.game - b.game)}));
          } else if (/^\/game\/\d+$/.test(url.pathname)) {
            const game = url.pathname.split('/').pop();
            const text = await readFile(`${directory}/game-${game}.jsonl`, 'utf8');
            const lines = text.slice(0, text.lastIndexOf('\n') + 1).trim().split('\n').filter(Boolean);
            const first = lines.length ? JSON.parse(lines[0]) : null;
            const sameGame = String(first?.seed) === url.searchParams.get('seed');
            const after = sameGame ? Number(url.searchParams.get('after') ?? -1) : -1;
            const rows = lines.slice(Number.isFinite(after) ? Math.max(0, after + 1) : 0).map(x => JSON.parse(x));
            res.end(JSON.stringify({seed: first?.seed, reset: !sameGame, rows}));
          } else { res.statusCode = 404; res.end('{}'); }
        } catch { res.statusCode = 503; res.end(JSON.stringify({error: 'Waiting for the next self-play game'})); }
      });
    },
  };
}
