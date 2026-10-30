import sys,unittest,torch,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from core import Model,make_data,reflect,rank1_trans,rotary,count_params,bytes_inference,P,TASKS,ARMS,POSITIONS
from torch.nn import functional as F

class TestCore(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1)
 def test_reflection_involution(self):
  x=torch.randn(4,7,16,dtype=torch.float64);v=torch.randn(16,dtype=torch.float64)
  self.assertLess((reflect(reflect(x,v),v)-x).abs().max().item(),1e-12)
 def test_rope_preserve_norm(self):
  x=torch.randn(3,11,16,dtype=torch.float64)
  self.assertLess((rotary(x).norm(dim=-1)-x.norm(dim=-1)).abs().max().item(),1e-12)
 def test_coupled_rank1_exact_mirror_at_beta_minus2(self):
  x=torch.randn(5,8,16,dtype=torch.float64);v=torch.randn(16,dtype=torch.float64)
  a=reflect(F.silu(reflect(x,v)),v)
  b=rank1_trans(F.silu(rank1_trans(x,v,-2.0)),v,-2.0)
  self.assertLess((a-b).abs().max().item(),1e-12)
 def test_matched_backbone(self):
  models=[Model(a,p,211) for a in ARMS for p in POSITIONS]
  for name in ['emb','block','norm','head']:
   orig=getattr(models[0],name).state_dict()
   for model in models:
    self.assertTrue(all(torch.equal(orig[k],v) for k,v in getattr(model,name).state_dict().items()))
 def test_arms_params_and_bytes(self):
  m={a:Model(a,'hybrid',101) for a in ARMS}
  self.assertEqual(count_params(m['mirror']),count_params(m['film']))
  self.assertEqual(count_params(m['coupled_rank1'])-count_params(m['mirror']),2)
  self.assertGreater(bytes_inference(m['mirror']),0)
 def test_rope_hybrid_nope_head_map(self):
  # For hybrid, head0 uses rotary and head1 uses raw Q/K. NoPE head0 and head1 are unmodified.
  for p in POSITIONS:
   m=Model('mirror',p,101)
   x=torch.randint(0,8,(3,11));x[:,-1]=8
   self.assertEqual(m(x).shape,(3,2))
 def test_data_labels(self):
  for task in TASKS:
   x,y=make_data(task,101,512,1)
   self.assertEqual(x.shape,(512,11));self.assertTrue((y>=0).all() and (y<=1).all())
   for xx,yy in zip(x.tolist(),y.tolist()):
    a=xx[:-1];query=xx[-1]
    if task=='set_membership':
     calc=int(any(z//2==query-8 for z in a))
    elif task=='last_value':calc=a[-1]%2
    else:
     matches=[z%2 for z in a if z//2==query-8]
     self.assertEqual(len(matches),2);self.assertNotEqual(matches[0],matches[1]);calc=matches[-1]
    self.assertEqual(calc,yy)
 def test_split_separation_and_ood(self):
  a,b=make_data('last_matching_key',101,64,1)
  c,d=make_data('last_matching_key',101,64,2)
  self.assertFalse(torch.equal(a,c))
  o,oy=make_data('last_matching_key',101,64,4,content_length=14)
  self.assertEqual(o.shape,(64,15))
 def test_all_arms_finite_gradients(self):
  x,y=make_data('last_matching_key',101,12,1)
  for arm in ARMS:
   m=Model(arm,'hybrid',101);loss=F.cross_entropy(m(x),y);loss.backward()
   self.assertTrue(torch.isfinite(loss).item())
   self.assertTrue(all(torch.isfinite(p.grad).all().item() for p in m.parameters() if p.grad is not None))
if __name__=='__main__':unittest.main()