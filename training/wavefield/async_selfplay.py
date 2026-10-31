"""Persistent CPU actors with bounded, durable delivery to the learner.

A game is acknowledged only after its samples and ID enter a durable learner
checkpoint. Unacknowledged games are replayed after a restart.
"""
from __future__ import annotations

import atexit
import json
import multiprocessing as mp
import os
import queue
import time
from pathlib import Path

import torch

from .engine import RustEngine
from .model import PolicyValueNet
from .encoding import SIDE_SIZE, board_channels_for_view
from .mcts import mcts_selfplay_records
from .spectator import SpectatorFeed


def atomic_save(value, path):
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    with temporary.open('wb') as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _actor(slot, directory, settings, permits, counter, counter_lock, results, stop):
    torch.set_num_threads(1)
    os.nice(5)
    engine = RustEngine(retain_history=False)
    model = None
    version = None
    try:
        while not stop.is_set():
            if not permits.acquire(timeout=.5):
                continue
            if stop.is_set():
                permits.release()
                break
            with counter_lock:
                game_id = counter.value
                counter.value += 1
                path = directory / 'counter.json'
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps({'next_game_id': counter.value}))
                temporary.replace(path)
            # Atomic replacement means this open file holds one complete version.
            with (directory / 'model.pt').open('rb') as handle:
                snapshot = torch.load(handle, map_location='cpu', weights_only=False)
            if version != snapshot['model_version']:
                if model is None:
                    model = PolicyValueNet(**snapshot['model_config'])
                model.load_state_dict(snapshot['model'])
                model.eval()
                version = snapshot['model_version']
            source_iteration = snapshot['iteration']
            del snapshot
            seed = settings['seed'] + 60_000 + game_id
            started = time.perf_counter()
            with SpectatorFeed(slot, seed, settings['config'].simulations,
                               settings['max_plies'], settings['config'].max_tuning_actions,
                               settings['config'].value_discount, iteration=source_iteration,
                               game_id=game_id, model_version=version) as feed:
                record = mcts_selfplay_records(
                    engine, model, games=1, max_plies=settings['max_plies'],
                    seed=seed, config=settings['config'], device='cpu',
                    input_view=settings['input_view'], history_plies=settings['history_plies'],
                    cap_value=settings['cap_value'], on_position=feed.publish)[0]
            metadata = dict(game_id=game_id, worker=slot, model_version=version,
                            source_iteration=source_iteration,
                            seconds=time.perf_counter() - started)
            for sample in record.samples:
                sample.metadata.update(source_model_version=version, game_id=game_id)
            atomic_save(dict(record=record, **metadata), directory / 'inbox' / f'game-{game_id:012d}.pt')
            results.put(metadata)
    finally:
        engine.close()


class AsyncSelfplayPool:
    def __init__(self, model, args, config, version, iteration, restored=None, worker_target=_actor):
        self.directory = args.run_dir / 'async-selfplay'
        self.inbox = self.directory / 'inbox'
        self.inbox.mkdir(parents=True, exist_ok=True)
        # A previous process may have stopped during an atomic write.
        for temporary in self.inbox.glob('*.tmp'):
            temporary.unlink()
        self.consumed = set((restored or {}).get('consumed_game_ids', []))
        self.delivered = []
        self.recovered = []
        highest = iteration * args.mcts_games_per_iteration
        for path in sorted(self.inbox.glob('game-*.pt')):
            game_id = int(path.stem.split('-')[-1])
            highest = max(highest, game_id)
            if game_id in self.consumed:
                path.unlink()
            else:
                self.recovered.append(game_id)
        counter_path = self.directory / 'counter.json'
        saved_counter = json.loads(counter_path.read_text())['next_game_id'] if counter_path.exists() else 0
        next_id = max(highest + 1, saved_counter, (restored or {}).get('next_game_id', 0))
        capacity = args.async_queue_capacity or 3 * args.mcts_workers
        if capacity < args.mcts_workers + args.mcts_games_per_iteration:
            raise ValueError('Async capacity must accommodate the workers plus one learner cohort')
        self.capacity = max(capacity, len(self.recovered) + args.mcts_workers)
        ctx = mp.get_context('spawn')
        self.permits = ctx.BoundedSemaphore(self.capacity)
        for _ in self.recovered:
            self.permits.acquire()
        self.counter = ctx.Value('q', next_id)
        self.lock = ctx.Lock()
        self.results = ctx.Queue()
        self.stop = ctx.Event()
        self.model_config = dict(hidden_size=args.hidden_size,
                                 board_channels=board_channels_for_view(args.input_view), side_size=SIDE_SIZE,
                                 architecture=args.model_arch, history_plies=args.history_plies,
                                 encoder_arch=getattr(model, 'encoder_arch', None) or 'transformer',
                                 encoder_hidden_size=getattr(model, 'encoder_hidden_size', None) or args.hidden_size,
                                 freeze_encoder=getattr(model, 'freeze_encoder', True))
        self.publish(model, version, iteration)
        settings = dict(config=config, seed=args.seed, max_plies=args.max_plies,
                        input_view=args.input_view, history_plies=args.history_plies, cap_value=args.cap_value)
        self.workers = [ctx.Process(target=worker_target, args=(slot, self.directory, settings,
                        self.permits, self.counter, self.lock, self.results, self.stop),
                        name=f'selfplay-{slot}') for slot in range(args.mcts_workers)]
        for worker in self.workers:
            worker.start()
        atexit.register(self.close)

    def publish(self, model, version, iteration):
        atomic_save(dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                         model_config=self.model_config, model_version=version, iteration=iteration),
                    self.directory / 'model.pt')

    def get_batch(self, count):
        batch = []
        while len(batch) < count:
            if self.recovered:
                game_id = self.recovered.pop(0)
            else:
                try:
                    game_id = self.results.get(timeout=1)['game_id']
                except queue.Empty:
                    failed = [(p.name, p.exitcode) for p in self.workers if not p.is_alive()]
                    if failed:
                        raise RuntimeError(f'Self-play worker exited: {failed}; completed games are retained for recovery')
                    continue
            payload = torch.load(self.inbox / f'game-{game_id:012d}.pt', map_location='cpu', weights_only=False)
            batch.append(payload)
            self.delivered.append(game_id)
        return batch

    def checkpoint_state(self):
        return dict(consumed_game_ids=sorted(self.consumed.union(self.delivered)),
                    next_game_id=self.counter.value)

    def acknowledge(self):
        for game_id in self.delivered:
            self.consumed.add(game_id)
            (self.inbox / f'game-{game_id:012d}.pt').unlink(missing_ok=True)
            self.permits.release()
        self.delivered.clear()

    def close(self):
        self.stop.set()
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()
        for worker in self.workers:
            worker.join(timeout=3)
        atexit.unregister(self.close)
