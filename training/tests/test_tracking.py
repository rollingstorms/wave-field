from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from training.track_runs import event_metrics, read_complete_events


class TrackingTest(unittest.TestCase):
    def test_incomplete_tail_is_retried_without_duplicate_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            first = json.dumps({"event": "first"}) + "\n"
            path.write_text(first + '{"event":')
            rows, offset = read_complete_events(path, 0)
            self.assertEqual(rows, [{"event": "first"}])
            self.assertEqual(offset, len(first.encode()))
            with path.open("a") as handle:
                handle.write('"second"}\n')
            rows, offset = read_complete_events(path, offset)
            self.assertEqual(rows, [{"event": "second"}])
            self.assertEqual(offset, path.stat().st_size)

    def test_loss_steps_follow_optimizer_updates_not_search_simulations(self) -> None:
        state = {}
        event_metrics({"event": "config", "config": {"batch_size": 4}}, state)
        row = {"event": "train_epoch", "samples": 9, "seconds": 0.1,
               "loss": 2.0, "kind": 0.2, "policy": 1.0, "tuning": 0.3, "value": 0.5}
        metrics = event_metrics(row, state)
        self.assertEqual(state["optimizer_steps"], 3)
        self.assertEqual(metrics["loss/total"], 2.0)
        event_metrics(row, state)
        self.assertEqual(state["optimizer_steps"], 6)
        event_metrics({"event": "learning_update", "iteration": 4, "optimizer_steps_total": 24}, state)
        event_metrics({"event": "config", "config": {"batch_size": 4}}, state)
        self.assertEqual(state["optimizer_steps"], 24)

    def test_generation_and_evaluation_keep_capped_games_visible(self) -> None:
        state = {}
        metrics = event_metrics({"event": "generate", "seconds": 10,
            "summary": {"games": 8, "mean_plies": 400, "samples": 100,
                        "wins": {"red": 2, "blue": 2, "capped": 4},
                        "metadata": {"action_kinds": {"move": 80, "tune": 20}}}}, state)
        self.assertEqual(metrics["self_play/capped_rate"], 0.5)
        self.assertEqual(metrics["self_play/positions_per_second"], 10)
        self.assertEqual(metrics["self_play/tunes_per_move"], 0.25)
        metrics = event_metrics({"event": "checkpoint_eval", "games": 8,
            "score": 0.5, "wins": 2, "losses": 2, "capped": 4, "opponent_iteration": 10}, state)
        self.assertEqual(metrics["evaluation/capped_rate"], 0.5)
        self.assertEqual(metrics["evaluation/win_rate"], 0.25)


if __name__ == "__main__":
    unittest.main()
