"""Unit tests for intervention policy, independent of its learned quality."""
import sys,unittest
from pathlib import Path
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
import run
from model import new_model
from data import dataset,batch_schedule

torch.set_num_threads(1)
class GuardTests(unittest.TestCase):
 def setUp(self):
  self.m=new_model('mirror',21);self.opt=run.opt_new(self.m)
  self.train=dataset('aligned_sparse_mirror','train',1327,128)
  self.dev=dataset('aligned_sparse_mirror','dev',1327,48)
  self.sched=batch_schedule(128,1327,200,32)
  self.groups={'true_shared_gradient_conflict':(0,1,2,3),'low_gate':(4,5,6,7),
    'low_edge_vjp_norm':(8,9,10,11),'random':(12,13,14,15)}
  self.audit={'relative_vjp_error_max':1e-7,'negative_grad_pair_fraction':0.1,
     'anti_aligned_edge_count':1,'top_scores':[],'core_grad_norm':1.0}
 def run_mock(self,values):
  with patch.object(run,'intervention_groups',return_value=(self.groups,self.audit)),\
       patch.object(run,'probe',side_effect=[(v,100) for v in values]):
   return run.decide(self.m,self.opt,'mirror_guard_long','aligned_sparse_mirror',1327,0,20,self.train,self.dev,self.sched,1.0)
 def test_guard_refuses_every_bad_cut(self):
  before=self.m.active.clone()
  result=self.run_mock([.2,.201,.202,.2011,.203])
  self.assertIsNone(result['selected']);self.assertEqual(result['active_after'],32)
  self.assertTrue(torch.equal(self.m.active,before))
  self.assertEqual(result['lookahead_probe_updates'],200)
 def test_guard_accepts_only_strict_improvement(self):
  result=self.run_mock([.2,.205,.198,.199,.21])
  self.assertEqual(result['selected'],'low_gate')
  self.assertEqual(result['edges_cut'],[4,5,6,7])
  self.assertEqual(result['active_after'],28)
 def test_guard_rejects_insufficient_gain(self):
  result=self.run_mock([.2,.201,.1997,.21,.212])
  self.assertIsNone(result['selected'])

if __name__=='__main__':unittest.main()
