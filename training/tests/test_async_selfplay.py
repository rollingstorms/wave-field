import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from wavefield.async_selfplay import AsyncSelfplayPool, atomic_save
from wavefield.mcts import MctsConfig
from wavefield.model import PolicyValueNet


def fake_actor(slot, directory, settings, permits, counter, counter_lock, results, stop):
    while not stop.is_set():
        if not permits.acquire(timeout=.1):
            continue
        with counter_lock:
            game_id = counter.value
            counter.value += 1
            (directory / 'counter.json').write_text(json.dumps({'next_game_id': counter.value}))
        snapshot = torch.load(directory / 'model.pt', weights_only=False)
        if slot == 0:
            while not (directory / 'fast-started').exists() and not stop.is_set():
                time.sleep(.01)
            time.sleep(1.5)
        else:
            (directory / 'fast-started').touch()
            time.sleep(.03)
        payload = dict(game_id=game_id, worker=slot, model_version=snapshot['model_version'],
                       source_iteration=snapshot['iteration'], seconds=.03, record=None)
        atomic_save(payload, directory / 'inbox' / f'game-{game_id:012d}.pt')
        results.put({k: v for k, v in payload.items() if k != 'record'})


class AsyncSelfplayTests(unittest.TestCase):
    def test_streaming_bounded_queue_and_durable_recovery(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(run_dir=Path(temporary), mcts_games_per_iteration=3,
                                   mcts_workers=2, async_queue_capacity=6, hidden_size=16,
                                   input_view='base', model_arch='conv', history_plies=1,
                                   seed=123, max_plies=300, cap_value='position')
            model = PolicyValueNet(hidden_size=16)
            pool = AsyncSelfplayPool(model, args, MctsConfig(), 10, 1, worker_target=fake_actor)
            try:
                first = pool.get_batch(3)
                # A fast actor supplies a cohort without waiting for the slow one.
                self.assertEqual([x['worker'] for x in first], [1, 1, 1])
                pool.publish(model, 11, 2)
                time.sleep(.25)
                self.assertLessEqual(pool.counter.value - 4, pool.capacity)
                pool.close()
                # No checkpoint: delivered games must still be available on restart.
                resumed = AsyncSelfplayPool(model, args, MctsConfig(), 11, 1, worker_target=fake_actor)
                try:
                    recovered = resumed.get_batch(3)
                    self.assertEqual({x['game_id'] for x in recovered}, {x['game_id'] for x in first})
                    state = resumed.checkpoint_state()
                    # Simulate a crash after saving the learner but before deleting games.
                finally:
                    resumed.close()
                final = AsyncSelfplayPool(model, args, MctsConfig(), 12, 2, restored=state, worker_target=fake_actor)
                try:
                    for game_id in state['consumed_game_ids']:
                        self.assertFalse((final.inbox / f'game-{game_id:012d}.pt').exists())
                    next_batch = final.get_batch(3)
                    self.assertFalse(set(state['consumed_game_ids']) & {x['game_id'] for x in next_batch})
                    self.assertTrue(any(x['model_version'] >= 11 for x in next_batch))
                    final.acknowledge()
                    self.assertEqual(final.delivered, [])
                finally:
                    final.close()
            finally:
                pool.close()
