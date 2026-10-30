from __future__ import annotations

import unittest
import torch

from wavefield.engine import RustEngine
from wavefield.model import PolicyValueNet
from wavefield.mcts import MctsConfig, mcts_selfplay_records, parallel_mcts_selfplay_records
from wavefield.shared_inference import SharedInferenceClient, SharedInferencePool


class SharedInferenceTest(unittest.TestCase):
    def test_shared_cpu_returns_equivalent_logits(self):
        torch.set_num_threads(1)
        torch.manual_seed(321)
        model = PolicyValueNet(hidden_size=32).eval()
        board, side = torch.randn(1, 13, 7, 7), torch.randn(1, 15)
        with SharedInferencePool(model, 1, 'cpu', max_batch=4, wait_ms=0) as service:
            client = SharedInferenceClient(service.spec)
            with torch.no_grad():
                expected = model.search_outputs(board, side)
            actual = client.search_outputs(board, side)
            for x, y in zip(expected, actual):
                torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)
        self.assertEqual(service.stats['positions'], 1)
        self.assertEqual(service.stats['batches'], 1)

    def test_shared_selfplay_preserves_cpu_search_and_targets(self):
        torch.set_num_threads(1)
        torch.manual_seed(52)
        model = PolicyValueNet(hidden_size=32)
        config = MctsConfig(value_discount=.997, simulations=2, top_k=4, max_depth=2, max_tuning_actions=1)
        engine = RustEngine(retain_history=False)
        stats = {}
        try:
            serial = mcts_selfplay_records(engine, model, games=2, max_plies=2, seed=53, config=config)
            parallel = parallel_mcts_selfplay_records(model, games=2, workers=2, max_plies=2,
                seed=53, config=config, inference_device='cpu', inference_max_batch=2,
                inference_wait_ms=2, inference_stats=stats)
            for expected, actual in zip(serial, parallel):
                self.assertEqual(expected.final_state, actual.final_state)
                self.assertEqual(len(expected.samples), len(actual.samples))
                for a, b in zip(expected.samples, actual.samples):
                    self.assertEqual(a.action_index, b.action_index)
                    self.assertEqual(a.action_kind, b.action_kind)
                    self.assertAlmostEqual(a.value, b.value)
        finally:
            engine.close()
        self.assertGreater(stats['positions'], 0)
        self.assertNotIn('error', stats)


if __name__ == '__main__':
    unittest.main()
