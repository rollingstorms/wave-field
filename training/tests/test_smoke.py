from __future__ import annotations

import unittest

import torch

from wavefield.encoding import ACTION_SIZE, BOARD_CHANNELS, SIDE_SIZE, decode_action, encode_state
from wavefield.engine import RustEngine, load_initial_state
from wavefield.eval import aggregate
from wavefield.model import PolicyValueNet, masked_policy_logits
from wavefield.selfplay import play_game, random_selfplay_game


class TrainingSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = RustEngine()
        cls.engine.build_release()

    def test_encode_initial_state_and_forward_pass(self) -> None:
        state = load_initial_state()
        actions = self.engine.legal_actions(state)
        board, side, legal_mask = encode_state(state, self.engine, actions)

        self.assertEqual(board.shape, (BOARD_CHANNELS, 7, 7))
        self.assertEqual(side.shape, (SIDE_SIZE,))
        self.assertEqual(legal_mask.shape, (ACTION_SIZE,))
        self.assertGreater(int(legal_mask.sum()), 0)

        model = PolicyValueNet(hidden_size=32)
        logits, value = model(
            torch.tensor(board).unsqueeze(0),
            torch.tensor(side).unsqueeze(0),
        )
        masked = masked_policy_logits(logits, torch.tensor(legal_mask).unsqueeze(0))
        self.assertEqual(logits.shape, (1, ACTION_SIZE))
        self.assertEqual(value.shape, (1,))
        self.assertTrue(torch.isfinite(masked.max()))

    def test_decode_action_round_trip_shape(self) -> None:
        action = decode_action(0)
        self.assertEqual(action["pieceId"], "blue-rook-1")
        self.assertEqual(action["destination"], {"x": 0, "y": 0})

    def test_random_selfplay_generates_samples(self) -> None:
        samples = random_selfplay_game(self.engine, max_plies=4, seed=17)
        self.assertGreater(len(samples), 0)
        self.assertEqual(samples[0].board.shape, (BOARD_CHANNELS, 7, 7))

    def test_model_selfplay_generates_stats(self) -> None:
        model = PolicyValueNet(hidden_size=32)
        record = play_game(self.engine, max_plies=3, seed=23, policy="model", model=model)
        self.assertGreater(len(record.samples), 0)
        self.assertEqual(record.stats.plies, len(record.samples))
        summary = aggregate([record])
        self.assertEqual(summary["games"], 1)
        self.assertIn("avg_pressure", summary)


if __name__ == "__main__":
    unittest.main()
