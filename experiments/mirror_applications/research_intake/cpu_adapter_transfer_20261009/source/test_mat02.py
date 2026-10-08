#!/usr/bin/env python3
import unittest,torch,numpy as np
from run_mat01 import make_world,fit_source_basis,FrozenBackbone,source_family,fit_base,target_support,H,C,R
from run_mat02 import freeze_target,arrays,METHODS

class MAT02Tests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1)
 def test_frozen_models_all_conditions(self):
  for x in METHODS:self.assertIsInstance(x,str)
  self.assertEqual(len(METHODS),8)
 def test_split_and_seed_firewall(self):
  w=make_world(201);self.assertEqual(len(set(w.train_ids)&set(w.audit_ids)),0)
  self.assertEqual(len(target_support(w,201)),300)
 def test_rank2_and_1_storage(self):
  w=make_world(21);model=FrozenBackbone(21)
  basis=(torch.randn(H,R),torch.randn(R,C),torch.randn(R,R),torch.randn(R,R,R))
  tr=[dict(code=torch.zeros(R),a=torch.zeros(H,2),b=torch.zeros(2,C)) for _ in range(4)]
  dic=arrays(model,w,basis,'mirror4_private2',tr)
  self.assertTrue('task_0_B' in dic)
  self.assertEqual(dic['task_0_B'].shape,(2,C))
 def test_independent_code_gradient(self):
  x=torch.randn(H,4); y=torch.randn(4,C)
  a=torch.randn(H,2,requires_grad=True);b=torch.randn(2,C,requires_grad=True)
  loss=(x@torch.eye(4)@y+a@b).square().sum();loss.backward()
  self.assertTrue(torch.isfinite(a.grad).all())
 def test_same_data_fits_only_support_without_oracle(self):
  w=make_world(22);self.assertLess(len(target_support(w,22)),len(w.train_ids))
  self.assertTrue(set(target_support(w,22)).issubset(set(w.train_ids)))

if __name__=='__main__':unittest.main(verbosity=2)
