import sys,unittest,math
from pathlib import Path
import torch
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/'source'))
from model import D,EDGES,VirtualGraph,new_model
from data import dataset,batch_schedule
from run import core_vjp_per_edge,intervention_groups,core_rotation,model_cost,probe,opt_new,nmse_loss,train_data,CONFIG

torch.set_num_threads(1)
class ExperimentTests(unittest.TestCase):
 def test_canonical_dimensions_and_weights(self):
  m=new_model('mirror',19)
  self.assertEqual(EDGES,32)
  self.assertEqual(sum(p.numel() for p in m.parameters()),320)
  self.assertEqual(tuple(m.weights().shape),(32,2,8,8))
  self.assertEqual(tuple(m(torch.zeros((3,D))).shape),(3,D))
 def test_cut_skips_and_packet_replay(self):
  m=new_model('mirror',3); before=model_cost(m)
  m.cut((0,1,2,3)); self.assertEqual(m.active_count(),28)
  self.assertEqual(model_cost(m),before*28//32)
  n=new_model('mirror',7);n.reload_packet(m.packet())
  x=torch.randn(5,D);self.assertTrue(torch.allclose(n(x),m(x),atol=1e-6))
 def test_exact_shared_core_vjp_reconstructs(self):
  m=new_model('mirror',19)
  data=dataset('aligned_sparse_mirror','train',1327,24)
  ids,g,error=core_vjp_per_edge(m,data,torch.arange(12),1.0)
  self.assertEqual(len(ids),32)
  self.assertEqual(tuple(g.shape),(32,128))
  self.assertLess(error,1e-4)
 def test_true_vjp_conflict_not_gate_sign(self):
  m=new_model('mirror',19)
  pair=dataset('aligned_sparse_mirror','train',1327,128)
  sched=batch_schedule(128,1327,180,32)
  grp,audit=intervention_groups(m,pair,sched,20,1.0,'aligned_sparse_mirror',1327,0,True)
  self.assertEqual(len(grp),4)
  self.assertTrue(all(len(v)==4 for v in grp.values()))
  self.assertLess(audit['relative_vjp_error_max'],1e-4)
  self.assertTrue(math.isfinite(audit['negative_grad_pair_fraction']))
 def test_keep_probe_is_immutable_and_guard(self):
  m=new_model('mirror',24);opt=opt_new(m)
  pair=dataset('independent_native_mlp','train',1327,128)
  dev=dataset('independent_native_mlp','dev',1327,40)
  sched=batch_schedule(128,1327,160,16)
  state={k:v.clone() for k,v in m.state_dict().items()}
  x,c=probe(m,opt,(),pair,sched,5,dev,1.,6)
  y,d=probe(m,opt,(),pair,sched,5,dev,1.,6)
  self.assertEqual(x,y);self.assertEqual(c,d)
  self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in state.items()))
 def test_gradient_changes_when_real_cut(self):
  m=new_model('mirror',19)
  pair=dataset('aligned_sparse_mirror','train',1327,128)
  a=core_rotation(m,[0,1,2,3],pair,torch.arange(16),1.0)
  self.assertIsNotNone(a);self.assertGreater(a,0)
  self.assertEqual(m.active_count(),32)
 def test_train_never_reads_fresh(self):
  with self.assertRaises(AssertionError):train_data('aligned_sparse_mirror','fresh',1327)

if __name__=='__main__':unittest.main()