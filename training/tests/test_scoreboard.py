import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve_selfplay import TrainingScoreboard


class ScoreboardTest(unittest.TestCase):
    def test_eta_includes_generation_training_eval_and_runtime_excludes_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stats = TrainingScoreboard(root/'status.json')
            stamp = lambda value: datetime.fromtimestamp(value, timezone.utc).isoformat()
            (root/'service-events.jsonl').write_text('\n'.join(json.dumps(dict(
                event='resume_after_inactivity', at=stamp(at), seconds=seconds)) for at, seconds in [(900, 200), (1500, 100)]))
            service = dict(active_effective_batch_size=4096, target_large_batch_updates=100,
                           large_batch_updates=10, last_start_at=stamp(1000), heartbeat=1990)
            for iteration in range(1, 6):
                stats.timings[iteration] = dict(complete=True, effective_batch_size=4096,
                    generation=60, epochs={epoch: dict(seconds=10, steps=1) for epoch in range(1, 4)},
                    evaluation={'checkpoint': 5} if iteration == 5 else {})
            # Exclude an unfinished iteration and an older microbatch configuration.
            stats.timings[6] = dict(effective_batch_size=4096, generation=1000)
            stats.timings[0] = dict(complete=True, effective_batch_size=128, generation=1000)
            with patch('serve_selfplay.time.time', return_value=2000):
                result = stats.time_estimate(service)
            self.assertEqual(result['batches'], 5)
            self.assertAlmostEqual(result['remaining_seconds'], 90 * 455 / 15)
            self.assertEqual(result['running_seconds'], 890)
            stats.timings.pop(1)
            self.assertIsNone(stats.time_estimate(service)['remaining_seconds'])
            service['large_batch_updates'] = 100
            self.assertEqual(stats.time_estimate(service)['remaining_seconds'], 0)

    def test_tactical_coverage_and_rates_weight_by_opportunities(self):
        common = dict(games=8, seconds=80, samples=100, plies=400, repetitions=0,
                      wins=dict(red=1, blue=1), actions=dict(move=80, tune=20))
        rows = [dict(common, iteration=40, tactics=None),
                dict(common, iteration=41, tactics=dict(games=8, rescues=1, rescue_opportunities=2,
                    rescues_by_player=dict(red=1, blue=0), opportunities_by_player=dict(red=2, blue=0),
                    pieces_lost=dict(red=8, blue=16), turns=400, tuning_turns=80,
                    tuning_actions=100, net_tuning_changes=50, decisive_games=2, decisive_plies=30)),
                dict(common, iteration=42, tactics=dict(games=8, rescues=0, rescue_opportunities=8,
                    pieces_lost=dict(red=0, blue=0), turns=400, tuning_turns=0))]
        tactics = TrainingScoreboard.summarize(rows)['tactics']
        self.assertEqual(tactics['games'], 16)
        self.assertEqual(tactics['first_iteration'], 41)
        self.assertEqual(tactics['rescue_rate'], .1)
        self.assertEqual(tactics['pieces_lost_per_game'], dict(red=.5, blue=1))
        self.assertEqual(tactics['tuning_turn_rate'], .1)
        self.assertEqual(tactics['tuning_change_rate'], .5)
        self.assertEqual(tactics['decisive_mean_plies'], 15)
        self.assertIsNone(TrainingScoreboard.summarize(rows[:1])['tactics']['rescue_rate'])

    def test_eta_ignores_old_selfplay_batch_sizes(self):
        scoreboard=TrainingScoreboard(Path('/tmp/unused-wavefield-status.json'))
        scoreboard.timings={i:dict(complete=True,effective_batch_size=4096,games=8,generation=60,
                                   epochs={1:dict(seconds=3,steps=1)}) for i in range(1,7)}
        service=dict(selfplay_games=128,active_effective_batch_size=4096,large_batch_updates=10,
                     target_large_batch_updates=100)
        self.assertIsNone(scoreboard.time_estimate(service)['remaining_seconds'])
        for i in range(7,12):
            scoreboard.timings[i]=dict(complete=True,effective_batch_size=4096,games=128,generation=600,
                                      epochs={1:dict(seconds=3,steps=1)})
        self.assertEqual(scoreboard.time_estimate(service)['remaining_seconds'],90*603)

    def test_rules_window_recovery_duplicates_and_incomplete_tail(self):
        def generation(iteration, red, blue, capped, repetition):
            return dict(event='generate', phase='mcts_model_session', iteration=iteration, seconds=80,
                summary=dict(games=8, samples=100, mean_plies=50, repetition_draws=repetition,
                    wins=dict(red=red, blue=blue, capped=capped),
                    metadata=dict(action_kinds=dict(move=80, tune=20))))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status = root/'status.json'
            status.write_text(json.dumps(dict(run_dir=tmp)))
            events = root/'events.jsonl'
            rows = [generation(40, 8, 0, 0, 0),
                    dict(event='learning_update', iteration=40),
                    dict(event='config', config=dict(mcts_repetition_limit=3)),
                    generation(41, 0, 0, 0, 8), generation(41, 1, 1, 1, 5),
                    generation(42, 0, 2, 1, 5),
                    dict(event='train_epoch', iteration=42, epoch=3, loss=3.2, value=.04,
                         validation_loss=3.3, validation_value=.05, validation_samples=100)]
            events.write_text(''.join(json.dumps(row)+'\n' for row in rows)+'{"event":')
            stats = TrainingScoreboard(status)
            result = stats.snapshot()
            self.assertEqual(result['rules_start'], 41)
            window = result['current']
            self.assertEqual((window['games'], window['red_wins'], window['blue_wins'], window['draws']), (16, 1, 3, 12))
            self.assertEqual(window['repetition_draws'], 10)
            self.assertEqual(window['mean_plies'], 50)
            self.assertEqual(window['games_per_hour'], 360)
            self.assertEqual(window['tunes_per_move'], .25)
            self.assertEqual(result['losses']['validation_loss'], 3.3)
            with events.open('a') as handle:
                handle.write('"learning_update","iteration":42}\n')
                for iteration in range(43, 65):
                    handle.write(json.dumps(generation(iteration, 1, 0, 0, 7))+'\n')
            stats.cached = None
            result = stats.snapshot()
            self.assertEqual(result['recent']['batches'], 20)
            self.assertEqual(result['recent']['first_iteration'], 45)
            self.assertEqual(result['current']['batches'], 24)
            self.assertEqual(stats.completed, 42)


if __name__ == '__main__':
    unittest.main()
