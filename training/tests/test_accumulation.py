from __future__ import annotations

import copy
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from wavefield.encoding import ACTION_SIZE, TUNING_ACTION_SIZE
from wavefield.model import PolicyValueNet
from wavefield.selfplay import Sample
from wavefield.train import samples_to_tensors, train_epoch, evaluate_loss
from wavefield.experiment import save_checkpoint, partition_validation_games
from wavefield.replay import ReplayBuffer
from types import SimpleNamespace


class AccumulationTest(unittest.TestCase):
    def test_accumulation_matches_full_batch_with_mixed_heads_and_partial_group(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        rng = np.random.default_rng(17)
        samples=[]
        for i in range(11):
            tune = i in (1, 2, 7)
            samples.append(Sample(board=rng.normal(size=(13,7,7)).astype(np.float32),
                side=rng.normal(size=15).astype(np.float32), legal_mask=np.ones(ACTION_SIZE,np.float32),
                action_index=-100 if tune else i, player='blue', value=float(i/11), action_kind=int(tune),
                legal_tuning_mask=np.ones(TUNING_ACTION_SIZE,np.float32), tuning_action_index=i if tune else -100))
        tensors=samples_to_tensors(samples,torch.device('cpu'))
        evaluation_model = PolicyValueNet(hidden_size=16)
        before = copy.deepcopy(evaluation_model.state_dict())
        full_validation = evaluate_loss(evaluation_model, tensors, batch_size=11)
        micro_validation = evaluate_loss(evaluation_model, tensors, batch_size=3)
        for key in full_validation:
            self.assertAlmostEqual(full_validation[key], micro_validation[key], places=6)
        self.assertTrue(evaluation_model.training)
        self.assertTrue(all(parameter.grad is None for parameter in evaluation_model.parameters()))
        for key, value in evaluation_model.state_dict().items():
            torch.testing.assert_close(value, before[key])
        a=PolicyValueNet(hidden_size=16);b=copy.deepcopy(a)
        opt_a=torch.optim.SGD(a.parameters(),lr=.01);opt_b=torch.optim.SGD(b.parameters(),lr=.01)
        torch.manual_seed(55);full=train_epoch(a,opt_a,tensors,batch_size=8,effective_batch_size=8)
        torch.manual_seed(55);micro=train_epoch(b,opt_b,tensors,batch_size=3,effective_batch_size=8)
        self.assertEqual(full['optimizer_steps'],2)
        self.assertEqual(micro['optimizer_steps'],2)
        for x,y in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(x,y,atol=1e-7,rtol=1e-6)
        self.assertAlmostEqual(full['loss'],micro['loss'],places=6)
        result=train_epoch(b,opt_b,tensors,batch_size=3,effective_batch_size=8,max_optimizer_steps=1)
        self.assertEqual(result['optimizer_steps'],1)
        self.assertEqual(result['samples_processed'],8)

    def test_recovery_rotation_retains_previous_and_prunes_only_milestones(self):
        torch.set_num_threads(1)
        model=PolicyValueNet(hidden_size=16);opt=torch.optim.AdamW(model.parameters())
        class Logger:
            def write(self,row):pass
        with tempfile.TemporaryDirectory() as tmp:
            args=Namespace(run_dir=Path(tmp),hidden_size=16,input_view='base',history_plies=1,
                           model_arch='conv',unfreeze_encoder=False,recovery_checkpoints=True,
                           large_batch_start_step=0,milestone_every_updates=2,keep_milestones=2)
            validation = ReplayBuffer(2)
            validation.extend(['heldout-1', 'heldout-2', 'heldout-3'])
            for iteration in range(1,7):
                save_checkpoint(model,opt,args,Logger(),iteration,10,optimizer_steps=iteration, validation_replay=validation)
            latest=torch.load(Path(tmp)/'checkpoint-latest.pt',weights_only=False)
            previous=torch.load(Path(tmp)/'checkpoint-previous.pt',weights_only=False)
            self.assertEqual(latest['optimizer_steps'],6)
            self.assertEqual(previous['optimizer_steps'],5)
            self.assertIsNotNone(latest['torch_rng_state'])
            restored = ReplayBuffer(2)
            restored.load_state_dict(latest['validation_replay'])
            self.assertEqual(list(restored.samples), ['heldout-2', 'heldout-3'])
            self.assertEqual(restored.total_seen, 3)
            self.assertEqual([p.name for p in sorted(Path(tmp).glob('checkpoint-update-*.pt'))],
                             ['checkpoint-update-000000004.pt','checkpoint-update-000000006.pt'])
            self.assertEqual(json.loads((Path(tmp)/'checkpoint-latest.json').read_text())['optimizer_steps'],6)

    def test_validation_games_never_enter_training_replay(self):
        records = [SimpleNamespace(samples=[object(), object()]) for _ in range(8)]
        training, validation = partition_validation_games(records, 1, 9)
        replay = ReplayBuffer(100)
        replay.extend(training)
        self.assertEqual(validation, records[1].samples)
        self.assertEqual(len(training), 14)
        self.assertFalse({id(x) for x in validation} & {id(x) for x in replay.samples})
        _, next_validation = partition_validation_games(records, 1, 10)
        self.assertEqual(next_validation, records[2].samples)
        with self.assertRaises(ValueError):
            partition_validation_games(records, 8, 1)


if __name__=='__main__':unittest.main()
