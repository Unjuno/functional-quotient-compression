import json,importlib.util,unittest,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ma1018_model',ROOT/'source'/'model.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class QuantumMirrorTests(unittest.TestCase):
 def test_draw_and_protocol(self):
  d=json.loads((ROOT/'source/random_draw.json').read_text());p=json.loads((ROOT/'PROTOCOL.json').read_text())
  self.assertEqual(d['selected_id'],'MA-1018');self.assertEqual(p['experiment_id'],d['selected_id']);self.assertEqual(d['pool_size'],560)
 def test_zero_angle_skeleton_is_unitary(self):
  u=m.unitary([0.0]*72);self.assertAlmostEqual(m.unitary_metrics(u,u)[0],1.0,places=12)
  self.assertLess(float(abs(u.conj().T@u-np.eye(16)).max()),1e-12)
 def test_tensor_factor_roundtrip_shapes(self):
  import torch,numpy as np
  x=torch.randn(6,6,4,3,dtype=torch.float64);cores=m.fit_tt(x,2);b=m.tt_basis(cores)
  self.assertEqual(tuple(b.shape),(cores[0].shape[-1],6,4,3))
  factors=[torch.randn(d,2,dtype=torch.float64) for d in (6,6,4,3)]
  self.assertEqual(tuple(m.cp_basis(factors).shape),(2,6,4,3))
  _,lb,lc=m.fit_linear_basis(x[:6],8);self.assertEqual(tuple(lb.shape),(6,6,4,3));self.assertEqual(tuple(lc.shape),(6,6))
if __name__=='__main__':
 import numpy as np
 unittest.main()
