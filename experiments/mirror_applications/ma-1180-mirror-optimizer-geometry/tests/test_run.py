"""Fast source-level mechanics tests, run before training; no fresh samples."""
import importlib.util, pathlib,sys,unittest
import torch
P=pathlib.Path(__file__).resolve().parents[1]/'source'/'run.py'
spec=importlib.util.spec_from_file_location('ma1180',P);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
PARENT,_=mod.load_parent()

class Tests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1)
 def test_householder_scale_sign_invariance(self):
  self.assertLess(mod.audit_mechanism(PARENT)['scale_sign_invariance_max_abs_error_fp64'],1e-10)
 def test_householder_radial_gradient_orthogonality(self):
  self.assertLess(mod.audit_mechanism(PARENT)['radial_gradient_cosine_max_fp64'],1e-8)
 def test_fixed_radius_retraction(self):
  v=torch.randn(3,16);r=v.norm(dim=-1,keepdim=True);v*=4.;mod.renorm(v,r)
  self.assertLess((v.norm(dim=-1,keepdim=True)-r).abs().max().item(),1e-5)
 def test_tangent_projection(self):
  v=torch.randn(3,16,dtype=torch.float64);g=torch.randn(3,16,dtype=torch.float64)
  t=mod.tangent(v,g)
  self.assertLess((t*v).sum(-1).abs().max().item(),1e-12)
 def test_sphere_scalar_optimizer_keeps_initial_norm(self):
  v=torch.randn(3,16,requires_grad=True);before=v.detach().norm(dim=-1)
  o=mod.SphereScalarAdam(v)
  for _ in range(10):
   v.grad=torch.randn_like(v);o.step()
  self.assertLess((v.detach().norm(dim=-1)-before).abs().max().item(),1e-5)
 def test_sphere_momentum_optimizer_keeps_initial_norm(self):
  v=torch.randn(3,16,requires_grad=True);before=v.detach().norm(dim=-1)
  o=mod.SphereMomentumSGD(v)
  for _ in range(10):
   v.grad=torch.randn_like(v);o.step()
  self.assertLess((v.detach().norm(dim=-1)-before).abs().max().item(),1e-5)
 def test_same_mirror_initial_predictions(self):
  x=torch.randn(12,4);r=torch.randint(3,(12,));
  for _ in mod.ARMS[:-1]:
   a=PARENT.new_student('mirror',992)(x,r);b=PARENT.new_student('mirror',992)(x,r)
   self.assertTrue(torch.equal(a,b))
 def test_same_physical_parameter_count(self):
  for kind in ('mirror','film'):
   self.assertEqual(PARENT.count(PARENT.new_student(kind,4422)),196)
 def test_total_trial_count(self):self.assertEqual(len(mod.SEEDS)*len(mod.WORLDS)*len(mod.ARMS),120)
 def test_parent_teacher_checkpoint_hash(self):
  import json
  data=json.loads((P.parents[1]/'PROTOCOL.json').read_text())
  root=pathlib.Path('/mnt/data/MA1179_work/checkpoints')
  for k,hash_expected in data['control_teacher_provenance']['teacher_sha256'].items():
   self.assertEqual(mod.sha(root/('teachers_seed'+k+'.pt')),hash_expected)

if __name__=='__main__': unittest.main()
