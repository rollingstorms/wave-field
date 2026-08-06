from __future__ import annotations

from argparse import Namespace
import unittest

import numpy as np
import torch

from wavefield.encoding import (
    ACTION_SIZE,
    BOARD_CHANNELS,
    RICH_BOARD_CHANNELS,
    SIDE_SIZE,
    decode_action,
    encode_state,
)
from wavefield.analyze_model import effective_rank, fit_linear_probe
from wavefield.engine import RustEngine, load_initial_state
from wavefield.eval import aggregate
from wavefield.experiment import parse_weights, replay_weight_samples, sample_metadata_summary
from wavefield.match import play_match_game
from wavefield.model import PolicyValueNet, masked_policy_logits
from wavefield.scenarios import DEFAULT_SCENARIOS, build_scenario_states
from wavefield.serve_model import ModelMoveServer
from wavefield.selfplay import (
    Sample,
    batched_model_selfplay_records,
    play_game,
    random_selfplay_game,
    session_model_selfplay_records,
)


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

    def test_piece_identity_view_and_residual_forward_pass(self) -> None:
        state = load_initial_state()
        actions = self.engine.legal_actions(state)
        board, side, legal_mask = encode_state(state, self.engine, actions, input_view="piece_identity")

        self.assertEqual(board.shape, (RICH_BOARD_CHANNELS, 7, 7))
        self.assertEqual(int(board[BOARD_CHANNELS:].sum()), len(state["pieces"]))

        model = PolicyValueNet(
            hidden_size=32,
            board_channels=RICH_BOARD_CHANNELS,
            side_size=SIDE_SIZE,
            architecture="residual",
        )
        logits, value = model(
            torch.tensor(board).unsqueeze(0),
            torch.tensor(side).unsqueeze(0),
        )
        masked = masked_policy_logits(logits, torch.tensor(legal_mask).unsqueeze(0))
        self.assertEqual(logits.shape, (1, ACTION_SIZE))
        self.assertEqual(value.shape, (1,))
        self.assertTrue(torch.isfinite(masked.max()))

    def test_piece_identity_view_and_transformer_forward_pass(self) -> None:
        state = load_initial_state()
        actions = self.engine.legal_actions(state)
        board, side, legal_mask = encode_state(state, self.engine, actions, input_view="piece_identity")

        model = PolicyValueNet(
            hidden_size=32,
            board_channels=RICH_BOARD_CHANNELS,
            side_size=SIDE_SIZE,
            architecture="transformer",
        )
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
        self.assertIn("ply_distribution", summary)

    def test_head_to_head_match_generates_rich_stats(self) -> None:
        model = PolicyValueNet(hidden_size=32)
        record = play_match_game(
            self.engine,
            policies={"blue": "model", "red": "random"},
            model=model,
            device="cpu",
            max_plies=3,
            seed=27,
            temperature=0.0,
            input_view="base",
        )
        summary = aggregate([record])
        self.assertEqual(summary["games"], 1)
        self.assertIn("win_rates", summary)
        self.assertIn("avg_final_material_balance_red", summary)

    def test_batched_model_selfplay_generates_samples(self) -> None:
        model = PolicyValueNet(hidden_size=32)
        records = batched_model_selfplay_records(
            self.engine,
            model,
            games=2,
            max_plies=3,
            seed=29,
            batch_size=2,
        )
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record.samples for record in records))
        self.assertEqual(records[0].samples[0].board.shape, (BOARD_CHANNELS, 7, 7))

    def test_session_model_selfplay_matches_legacy_deterministic_rollout(self) -> None:
        torch.manual_seed(41)
        model = PolicyValueNet(hidden_size=32)
        legacy = batched_model_selfplay_records(
            self.engine,
            model,
            games=2,
            max_plies=4,
            seed=43,
            temperature=0.0,
            batch_size=2,
        )
        session = session_model_selfplay_records(
            self.engine,
            model,
            games=2,
            max_plies=4,
            seed=43,
            temperature=0.0,
            batch_size=2,
        )

        self.assertEqual(len(session), len(legacy))
        for session_record, legacy_record in zip(session, legacy):
            self.assertEqual(session_record.final_state, legacy_record.final_state)
            self.assertEqual(session_record.stats.plies, legacy_record.stats.plies)
            self.assertEqual(len(session_record.samples), len(legacy_record.samples))
            for session_sample, legacy_sample in zip(session_record.samples, legacy_record.samples):
                np.testing.assert_allclose(session_sample.board, legacy_sample.board)
                np.testing.assert_allclose(session_sample.side, legacy_sample.side)
                np.testing.assert_allclose(session_sample.legal_mask, legacy_sample.legal_mask)
                self.assertEqual(session_sample.action_index, legacy_sample.action_index)
                self.assertEqual(session_sample.player, legacy_sample.player)
        summary = aggregate(session)
        self.assertEqual(summary["games"], 2)
        self.assertIn("rescue_rate", summary)
        self.assertIn("losses_by_player", summary)

    def test_scenarios_are_playable_and_tag_session_samples(self) -> None:
        states = build_scenario_states(self.engine, DEFAULT_SCENARIOS, games=len(DEFAULT_SCENARIOS), seed=47)
        for state in states:
            self.assertEqual(state["status"], "playing")
            self.assertGreater(len(self.engine.legal_actions(state)), 0)
            self.assertIn("scenario", state["metadata"])

        model = PolicyValueNet(hidden_size=32)
        records = session_model_selfplay_records(
            self.engine,
            model,
            games=len(states),
            max_plies=2,
            seed=53,
            temperature=0.0,
            batch_size=len(states),
            initial_states=states,
        )
        tagged = [
            sample.metadata.get("scenario")
            for record in records
            for sample in record.samples
        ]
        self.assertTrue(tagged)
        self.assertTrue(set(tagged).issubset(set(DEFAULT_SCENARIOS)))

    def test_rust_training_batch_generates_encoded_samples(self) -> None:
        batch = self.engine.generate_random_training_batch(load_initial_state(), games=1, max_plies=2, seed=31)
        self.assertEqual(batch["summary"]["games"], 1)
        self.assertGreater(batch["summary"]["samples"], 0)
        sample = batch["samples"][0]
        self.assertEqual(len(sample["board"]), BOARD_CHANNELS * 7 * 7)
        self.assertEqual(len(sample["side"]), SIDE_SIZE)
        self.assertGreater(len(sample["legalActionIndexes"]), 0)
        self.assertGreaterEqual(sample["actionIndex"], 0)
        self.assertLess(sample["actionIndex"], ACTION_SIZE)

    def test_experiment_replay_weights_and_metadata_summary(self) -> None:
        sample = Sample(
            board=np.zeros((BOARD_CHANNELS, 7, 7), dtype=np.float32),
            side=np.zeros((SIDE_SIZE,), dtype=np.float32),
            legal_mask=np.zeros((ACTION_SIZE,), dtype=np.float32),
            action_index=0,
            player="red",
            value=0.5,
            metadata={
                "source": "rust_session_model",
                "phase": "endgame",
                "legal_count": 4,
                "material_balance_current": -2,
                "low_material": True,
            },
        )

        weighted = replay_weight_samples(
            [sample],
            source_weights=parse_weights("rust_session_model=2"),
            phase_weights=parse_weights("endgame=3"),
        )
        summary = sample_metadata_summary(weighted)

        self.assertEqual(len(weighted), 6)
        self.assertEqual(summary["sources"], {"rust_session_model": 6})
        self.assertEqual(summary["phases"], {"endgame": 6})
        self.assertEqual(summary["low_material"], 6)
        self.assertEqual(summary["legal_count"]["mean"], 4.0)

    def test_local_model_server_uses_model_tuning_head(self) -> None:
        server = ModelMoveServer.__new__(ModelMoveServer)
        server.engine = self.engine
        server.model = PolicyValueNet(hidden_size=32)
        server.device = torch.device("cpu")
        server.input_view = "base"
        server.args = Namespace(
            checkpoint="test-checkpoint.pt",
            temperature=0.0,
            tuning_temperature=0.0,
            max_tuning_actions=1,
            min_tuning_actions=1,
        )
        state = load_initial_state()

        response = server.move(state)

        actions = response["actions"]
        self.assertGreater(len(actions), 0)
        self.assertEqual(actions[0]["type"], "tune")
        self.assertEqual(actions[-1]["type"], "move")
        self.assertEqual(response["tuningActions"], 1)

    def test_analysis_rank_and_linear_probe_helpers(self) -> None:
        torch.manual_seed(61)
        base = torch.randn(80, 3)
        features = torch.cat([base, base[:, :1] * 0.5], dim=1)
        rank = effective_rank(features)
        self.assertEqual(rank["samples"], 80)
        self.assertEqual(rank["dimensions"], 4)
        self.assertGreater(rank["effective_rank"], 1.0)

        labels = (base[:, 0] > 0).long()
        result = fit_linear_probe(
            features,
            labels,
            task="binary",
            epochs=50,
            lr=1.0e-2,
            batch_size=16,
            seed=67,
        )
        self.assertEqual(result["status"], "ok")
        self.assertGreaterEqual(result["accuracy"], result["majority_baseline"])


if __name__ == "__main__":
    unittest.main()
