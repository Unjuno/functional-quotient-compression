import unittest,torch,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run import Model,reflect,rope,data
class TestMechanism(unittest.TestCase):
 def test_rope_preserves_norm(self):
  x=torch.randn(2,10,16,dtype=torch.float64)
  self.assertLess((x.norm(dim=-1)-rope(x).norm(dim=-1)).abs().max().item(),1e-12)
 def test_householder_involution(self):
  x=torch.randn(2,10,16,dtype=torch.float64);v=torch.randn(16,dtype=torch.float64)
  self.assertLess((reflect(reflect(x,v),v)-x).abs().max().item(),1e-12)
 def test_causes_output_dimensions(self):
  for pos in ['nope','rope']:
   for arm in ['shared','film_q','mirror_q','mirror_residual']:
    m=Model(arm,pos,123)
    self.assertEqual(m(torch.randint(0,13,(3,10))).shape,(3,2))
 def test_paired_mirror_nope_invariance(self):
  q=torch.randn(4,16,dtype=torch.float64);k=torch.randn(4,16,dtype=torch.float64)
  code=torch.randn(16,dtype=torch.float64)
  self.assertLess(((reflect(q,code)@reflect(k,code).T)-(q@k.T)).abs().max().item(),1e-12)
 def test_data_structure(self):
  for task in ['set_membership','last_slot']:
   x,y=data(task,101,256,1)
   self.assertTrue((x[:,-1]==12).all())
   target=(x[:,:-1]==3)
   calc=target.any(1) if task=='set_membership' else target[:,-1]
   self.assertTrue(torch.equal(calc.long(),y))
 def test_same_backbone_init(self):
  m1=Model('mirror_q','nope',101);m2=Model('film_q','rope',101)
  self.assertTrue(all(torch.equal(x,y) for (a,x),(b,y) in zip(m1.block.state_dict().items(),m2.block.state_dict().items())))
if __name__=='__main__':unittest.main()
