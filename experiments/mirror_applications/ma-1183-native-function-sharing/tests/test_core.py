import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import torch
from core import ConditionalMLP,reflect,rankone,Teacher,dataset,DIN,K
from engine import ARMS,ACTS,eval_loss
class Tests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1)
 def test_householder_orthogonal(self):
  z=torch.randn(200,12,dtype=torch.float64);u=torch.randn(200,12,dtype=torch.float64)
  self.assertLess((reflect(reflect(z,u),u)-z).abs().max().item(),1e-12)
 def test_coupled_exact_mirror(self):
  for act in ACTS:
   a=ConditionalMLP('mirror',act,1);b=ConditionalMLP('coupled',act,1)
   for name in ('up','down','ub','db','code'):
    self.assertTrue(torch.equal(getattr(a,name),getattr(b,name)),name)
   x=torch.randn(100,DIN);t=torch.randint(K,(100,))
   self.assertLess((a(x,t)-b(x,t)).abs().max().item(),1e-6)
 def test_backbone_shared_init(self):
  a=ConditionalMLP('film','relu',5);b=ConditionalMLP('mirror','relu',5)
  for name in ('up','down','ub','db'):self.assertTrue(torch.equal(getattr(a,name),getattr(b,name)))
 def test_positive_teacher_no_leak(self):
  teacher=Teacher(3,'relu',0.);a=dataset(teacher,3,'train',32);b=dataset(teacher,3,'dev',32)
  self.assertEqual(a[0].shape,(K*32,DIN));self.assertFalse(torch.equal(a[0],b[0]))
  self.assertTrue(torch.isfinite(a[2]).all())
 def test_all_output_shapes(self):
  for arm in ARMS:
   a=ConditionalMLP(arm,'silu',3)
   self.assertEqual(a(torch.randn(11,DIN),torch.randint(K,(11,))).shape,(11,3))
 def test_budget(self):
  a={arm:ConditionalMLP(arm,'relu',1).physical_params() for arm in ARMS}
  self.assertEqual(a['mirror'],a['film']);self.assertEqual(a['coupled'],a['mirror']+K)
  self.assertTrue(abs(a['ind_budget']-a['mirror'])/a['mirror']<.05)
 def test_known_teacher_oracle(self):
  teacher=Teacher(13,'relu',0.)
  model=ConditionalMLP('mirror','relu',1)
  with torch.no_grad():
   model.up.copy_(teacher.a);model.ub.copy_(teacher.ab)
   model.down.copy_(teacher.b/teacher.scale[:,None]);model.db.copy_((teacher.bb-teacher.mean)/teacher.scale)
   model.code.copy_(teacher.codes)
  ds=dataset(teacher,13,'dev',128)
  self.assertLess(eval_loss(model,ds),1e-10)
if __name__=='__main__':unittest.main()
