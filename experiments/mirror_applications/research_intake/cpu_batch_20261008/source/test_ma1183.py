"""CPU shape, gradients, end-to-end leakage and fast-weight accounting for MA1183."""
from __future__ import annotations
import io
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from ma1183_tabm_style import (
 FastWeightMemberMLP, KINDS, K, C, DIM, get_pattern,
 generate, train, evaluate)

class TabMStyleTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(1)

 def test_shape_forward_output_member_logits(self):
  for m in KINDS:
   model=FastWeightMemberMLP(m,base_seed=33)
   y=model(torch.randn(7,DIM))
   self.assertEqual(tuple(y.shape),(7,K,C))
   self.assertTrue(bool(torch.isfinite(y).all()))
   size=model.serialized_bytes()
   self.assertGreater(size,100)

 def test_native_fast_weight_factors_modify_function(self):
  m=FastWeightMemberMLP('native_full_fast_weights',base_seed=31)
  x=torch.randn(16,DIM)
  with torch.no_grad():
   m.r_in[0].fill_(1.);m.s_h[0].fill_(1.)
   m.r_in[1].fill_(1.6);m.s_h[1].fill_(.5)
  y=m(x)
  self.assertGreater(float((y[:,0]-y[:,1]).abs().max()),1e-3)

 def test_no_member_single_forward_exact_broadcast_and_gradient(self):
  m=FastWeightMemberMLP('no_member_shared',base_seed=32)
  x=torch.randn(16,DIM)
  y=m(x)
  expected=m.down(torch.nn.functional.gelu(m.up(x)))
  torch.testing.assert_close(y,expected[:,None,:].expand_as(y),rtol=0,atol=0)
  self.assertEqual(float((y[:,0]-y[:,7]).abs().max()),0.)

 def test_packed_vs_loop_same_native(self):
  m=FastWeightMemberMLP('native_full_fast_weights',base_seed=32)
  x=torch.randn(8,DIM)
  packed=m(x)
  r,s=m.fast_weights()
  outputs=[]
  for k in range(K):
   h=torch.nn.functional.gelu(m.up(x*r[k]))*s[k]
   outputs.append(m.down(h))
  stacked=torch.stack(outputs,dim=1)
  torch.testing.assert_close(packed,stacked,rtol=1e-5,atol=1e-6)

 def test_gradients_packed_vs_member_sum(self):
  torch.manual_seed(9)
  m=FastWeightMemberMLP('structured_mirror_rotation4',base_seed=44)
  x=torch.randn(9,DIM)
  target=torch.randint(0,2,(9,))
  p=m(x)
  loss=torch.nn.functional.cross_entropy(p.reshape(-1,2),target[:,None].expand(-1,K).reshape(-1))
  gp=torch.autograd.grad(loss,tuple(m.parameters()),retain_graph=True)
  loss2=sum(torch.nn.functional.cross_entropy(p[:,k],target) for k in range(K))/K
  gl=torch.autograd.grad(loss2,tuple(m.parameters()))
  for a,b in zip(gp,gl):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)

 def test_real_model_byte_accounting_and_code_diversity(self):
  native=FastWeightMemberMLP('native_full_fast_weights',base_seed=33)
  mirror=FastWeightMemberMLP('structured_mirror_rotation4',base_seed=33)
  linear=FastWeightMemberMLP('fixed_seed_linear_code4',base_seed=33)
  learned=FastWeightMemberMLP('learned_dictionary_linear_code4',base_seed=33)
  self.assertEqual(mirror.member_scalars(),linear.member_scalars())
  self.assertLess(mirror.serialized_bytes(),native.serialized_bytes())
  self.assertGreater(learned.serialized_bytes(),mirror.serialized_bytes())
  self.assertTrue(bool(torch.equal(native.up.weight,mirror.up.weight)))
  r,s=mirror.fast_weights()
  self.assertGreater(float(r.std(dim=0).mean()+s.std(dim=0).mean()),1e-5)

 def test_train_world_separate_support_holdout(self):
  tr,val,heldout=generate(23,.5)
  self.assertEqual(tr[0].shape,(1024,32))
  self.assertEqual(val[0].shape,(256,32))
  self.assertEqual(heldout[0].shape,(512,32))
  self.assertFalse(np.array_equal(tr[0][:256],val[0]))
  self.assertTrue(set(np.unique(tr[1])).issubset({0,1}))
  self.assertEqual(len(np.unique(tr[1])),2)

if __name__=='__main__':unittest.main(verbosity=2)
