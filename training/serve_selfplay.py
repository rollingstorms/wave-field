"""Serve the built self-play viewer and narrow read-only local training APIs."""
from __future__ import annotations

import argparse
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import parse_qs, urlsplit

from service_common import read_json


class TrainingScoreboard:
    """Incremental, restart-aware summaries of completed self-play batches."""
    def __init__(self, status):
        self.status = status
        self.path = None
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.offset = 0
        self.generations = {}
        self.latest_loss = None
        self.evaluation = None
        self.completed = 0
        self.repetition = 0
        self.rules_start = 1
        self.cached_at = 0
        self.cached = None
        self.timings = {}

    def time_estimate(self, service):
        effective = service.get('active_effective_batch_size', service.get('effective_batch_size'))
        rows = [row for iteration, row in sorted(self.timings.items())
                if iteration >= self.rules_start and row.get('complete') and row.get('effective_batch_size') == effective][-20:]
        steps = sum(sum(epoch['steps'] for epoch in row.get('epochs', {}).values()) for row in rows)
        seconds = sum(row.get('generation', 0) + sum(epoch['seconds'] for epoch in row.get('epochs', {}).values())
                      + sum(row.get('evaluation', {}).values()) for row in rows)
        remaining = max(0, service.get('target_large_batch_updates', 100000) - service.get('large_batch_updates', 0))
        estimate = remaining * seconds / steps if len(rows) >= 5 and steps and seconds else None
        if remaining == 0:
            estimate = 0
        running = None
        try:
            start = datetime.fromisoformat(service['last_start_at']).timestamp()
            end = min(time.time(), service.get('heartbeat', time.time()))
            gaps = 0
            path = self.status.parent / 'service-events.jsonl'
            if path.exists():
                for line in path.read_text().splitlines():
                    try:
                        event = json.loads(line)
                        at = datetime.fromisoformat(event['at']).timestamp()
                        if event.get('event') == 'resume_after_inactivity' and start <= at <= end:
                            gaps += min(event.get('seconds', 0), at - start)
                    except (ValueError, KeyError, TypeError):
                        continue
            running = max(0, end - start - gaps)
        except (ValueError, KeyError, TypeError):
            pass
        return dict(remaining_seconds=estimate, running_seconds=running, batches=len(rows),
                    seconds_per_update=seconds / steps if steps else None)

    def apply(self, row):
        event = row.get('event')
        if event == 'config':
            repetition = row['config'].get('mcts_repetition_limit', 0)
            if repetition != self.repetition:
                self.rules_start = self.completed + 1
            self.repetition = repetition
        elif event == 'learning_update':
            self.completed = row['iteration']
            self.timings.setdefault(row['iteration'], {})['complete'] = True
        elif event == 'generate' and row.get('phase') == 'mcts_model_session':
            summary = row['summary']
            self.timings.setdefault(row['iteration'], {})['generation'] = row['seconds']
            # A recovered iteration replaces its prior entry instead of counting twice.
            self.generations[row['iteration']] = dict(iteration=row['iteration'], seconds=row['seconds'],
                games=summary['games'], samples=summary['samples'], plies=summary['mean_plies'] * summary['games'],
                wins=summary['wins'], repetitions=summary.get('repetition_draws', 0),
                tactics=summary.get('tactics'), legal_choices=summary.get('metadata', {}).get('legal_count', {}).get('mean'),
                actions=summary.get('metadata', {}).get('action_kinds', {}))
        elif event == 'train_epoch':
            timing = self.timings.setdefault(row['iteration'], {})
            timing['effective_batch_size'] = row.get('effective_batch_size')
            timing.setdefault('epochs', {})[row.get('epoch', 1)] = dict(seconds=row.get('seconds', 0), steps=row.get('optimizer_steps', 0))
            self.latest_loss = {key: row[key] for key in ('iteration', 'epoch', 'loss', 'value',
                'validation_loss', 'validation_value', 'validation_samples') if key in row}
        elif event == 'checkpoint_eval':
            self.timings.setdefault(row['iteration'], {}).setdefault('evaluation', {})['checkpoint'] = row.get('seconds', 0)
            self.evaluation = {key: row[key] for key in ('iteration', 'opponent_iteration', 'games',
                'wins', 'losses', 'capped', 'score', 'policy') if key in row}
        elif event in ('baseline_eval', 'tactical_eval', 'session_eval') and 'iteration' in row:
            self.timings.setdefault(row['iteration'], {}).setdefault('evaluation', {})[event + str(row.get('opponent', ''))] = row.get('seconds', 0)

    @staticmethod
    def summarize(rows):
        games = sum(row['games'] for row in rows)
        seconds = sum(row['seconds'] for row in rows)
        red = sum(row['wins'].get('red', 0) for row in rows)
        blue = sum(row['wins'].get('blue', 0) for row in rows)
        moves = sum(row['actions'].get('move', 0) for row in rows)
        measured = [row for row in rows if row.get('tactics') is not None]
        def total(key):
            return sum(row['tactics'].get(key, 0) for row in measured)
        def sides(key):
            return {player: sum(row['tactics'].get(key, {}).get(player, 0) for row in measured) for player in ('red', 'blue')}
        opportunities, rescues = total('rescue_opportunities'), total('rescues')
        measured_games = total('games')
        choices = [row for row in measured if row.get('legal_choices') is not None]
        choice_samples = sum(row['samples'] for row in choices)
        tactics = dict(games=measured_games, first_iteration=measured[0]['iteration'] if measured else None,
            rescues=rescues, rescue_opportunities=opportunities,
            rescue_rate=rescues/opportunities if opportunities else None,
            rescues_by_player=sides('rescues_by_player'), opportunities_by_player=sides('opportunities_by_player'),
            pieces_lost_per_game={player: value/measured_games if measured_games else None for player, value in sides('pieces_lost').items()},
            tuning_turn_rate=total('tuning_turns')/total('turns') if total('turns') else None,
            tuning_change_rate=total('net_tuning_changes')/total('tuning_actions') if total('tuning_actions') else None,
            decisive_mean_plies=total('decisive_plies')/total('decisive_games') if total('decisive_games') else None,
            mean_legal_choices=sum(row['legal_choices']*row['samples'] for row in choices)/choice_samples if choice_samples else None)
        return dict(games=games, batches=len(rows), red_wins=red, blue_wins=blue,
            tactics=tactics,
            draws=games-red-blue, repetition_draws=sum(row['repetitions'] for row in rows),
            capped=sum(row['wins'].get('capped', 0) for row in rows),
            decisive_rate=(red+blue)/games if games else None,
            mean_plies=sum(row['plies'] for row in rows)/games if games else None,
            batch_seconds=seconds/len(rows) if rows else None,
            games_per_hour=games/seconds*3600 if seconds else None,
            tunes_per_move=sum(row['actions'].get('tune', 0) for row in rows)/moves if moves else None,
            first_iteration=rows[0]['iteration'] if rows else None,
            last_iteration=rows[-1]['iteration'] if rows else None)

    def snapshot(self):
        with self.lock:
            service = read_json(self.status, {})
            if not service.get('run_dir'):
                return dict(available=False)
            path = Path(service['run_dir']) / 'events.jsonl'
            if path != self.path or (path.exists() and path.stat().st_size < self.offset):
                self.path = path
                self.reset()
            if self.cached and time.monotonic() - self.cached_at < 5:
                return self.cached
            with path.open('rb') as handle:
                handle.seek(self.offset)
                while True:
                    line = handle.readline()
                    if not line or not line.endswith(b'\n'):
                        break
                    self.offset = handle.tell()
                    try:
                        self.apply(json.loads(line))
                    except (ValueError, KeyError, TypeError):
                        continue
            rows = [row for iteration, row in sorted(self.generations.items()) if iteration >= self.rules_start]
            self.cached = dict(available=bool(rows), updated=time.time(), rules_start=self.rules_start,
                current=self.summarize(rows), recent=self.summarize(rows[-20:]),
                timing=self.time_estimate(service),
                losses=self.latest_loss, evaluation=self.evaluation)
            self.cached_at = time.monotonic()
            return self.cached


def create_handler(web_root: Path, feed: Path, status: Path):
    scoreboard = TrainingScoreboard(status)
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(web_root), **kwargs)
        def log_message(self, format, *args):
            if len(args) > 1 and str(args[1]) not in ('200', '304'):
                super().log_message(format, *args)
        def json_response(self, data, code=200):
            payload = json.dumps(data).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def do_GET(self):
            url = urlsplit(self.path)
            if not url.path.startswith('/__selfplay/'):
                if url.path == '/':
                    self.path = '/index.html'
                return super().do_GET()
            try:
                if url.path == '/__selfplay/games':
                    games = [read_json(path) for path in feed.glob('game-*.meta.json')]
                    games = sorted((game for game in games if game), key=lambda x: x['game'])
                    return self.json_response(dict(games=games, service=read_json(status, {})))
                if url.path == '/__selfplay/service':
                    return self.json_response(read_json(status, {}))
                if url.path == '/__selfplay/stats':
                    return self.json_response(scoreboard.snapshot())
                if re.fullmatch(r'/__selfplay/game/\d+', url.path):
                    game = int(url.path.rsplit('/', 1)[1])
                    text = (feed / f'game-{game}.jsonl').read_text()
                    lines = text[:text.rfind('\n') + 1].splitlines()
                    first = json.loads(lines[0]) if lines else {}
                    query = parse_qs(url.query)
                    same = str(first.get('seed')) == query.get('seed', [''])[0]
                    try:
                        after = max(-1, int(query.get('after', ['-1'])[0])) if same else -1
                    except ValueError:
                        after = -1
                    rows = [json.loads(line) for line in lines[after + 1:]]
                    return self.json_response(dict(seed=first.get('seed'), reset=not same, rows=rows))
                self.json_response(dict(error='Not found'), 404)
            except (OSError, ValueError):
                self.json_response(dict(error='Waiting for self-play data'), 503)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--web-root', type=Path, required=True)
    parser.add_argument('--feed-dir', type=Path, required=True)
    parser.add_argument('--status-file', type=Path, required=True)
    parser.add_argument('--port', type=int, default=5174)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('0.0.0.0', args.port), create_handler(args.web_root, args.feed_dir, args.status_file))
    print(f'Self-play viewer listening on port {args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
