import sys
import unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from model import SharedSNN, encode_rate_spikes, ENVELOPE, PHASES, TIMESTEPS, HIDDEN

class TaskTimeMirrorTests(unittest.TestCase):
    def test_each_condition_forward_backward(self):
        torch.set_num_threads(1)
        x=torch.rand(6,784); tasks=torch.tensor([0,1,2,3,0,2])
        spikes=encode_rate_spikes(x,tasks,123)
        self.assertEqual(tuple(spikes.shape),(6,TIMESTEPS,784))
        for condition in ('shared_no_task_gain','native_task_time_tebn','mirror_cyclic_phase','rank1_task_scalar_gate_on_shared_tebn_profile'):
            model=SharedSNN(condition)
            y,_=model(spikes,tasks)
            loss=y.square().mean(); loss.backward()
            self.assertEqual(tuple(y.shape),(6,10))
            self.assertTrue(torch.isfinite(y).all())
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_integer_phase_views_are_exact_cyclic_shifts(self):
        model=SharedSNN('mirror_cyclic_phase')
        with torch.no_grad(): model.log_gain_profile.copy_(torch.arange(TIMESTEPS*HIDDEN).reshape(TIMESTEPS,HIDDEN)/100)
        tasks=torch.arange(4)
        gain=model.gain(tasks)
        for task,phase in enumerate(PHASES.long().tolist()):
            indices=(torch.arange(TIMESTEPS)-phase)%TIMESTEPS
            expected=model.log_gain_profile[indices].clamp(-2.0,2.0).exp()
            self.assertTrue(torch.allclose(gain[task],expected,atol=1e-6))

    def test_mirror_and_rank1_have_equal_adaptation_state_size(self):
        mirror=SharedSNN('mirror_cyclic_phase')
        lowrank=SharedSNN('rank1_task_scalar_gate_on_shared_tebn_profile')
        m=sum(p.numel() for n,p in mirror.named_parameters() if 'input.' not in n and 'readout.' not in n)
        r=sum(p.numel() for n,p in lowrank.named_parameters() if 'input.' not in n and 'readout.' not in n)
        self.assertEqual(m,r)
        self.assertEqual(m,TIMESTEPS*HIDDEN+4)

    def test_phase_coordinate_has_finite_nonzero_gradient(self):
        model=SharedSNN('mirror_cyclic_phase')
        with torch.no_grad(): model.log_gain_profile.copy_(torch.sin(torch.arange(TIMESTEPS)[:,None].float()) * torch.ones(1,HIDDEN))
        tasks=torch.arange(4)
        weights=torch.arange(TIMESTEPS).float()[None,:,None].expand(4,-1,HIDDEN)
        (model.gain(tasks)*weights).sum().backward()
        self.assertIsNotNone(model.phase.grad)
        self.assertTrue(torch.isfinite(model.phase.grad).all())
        self.assertGreater(float(model.phase.grad.abs().sum()),0.0)

if __name__=='__main__': unittest.main()
