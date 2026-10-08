import sys,unittest
from pathlib import Path
import numpy as np
SOURCE=Path(__file__).resolve().parents[1]/'source';sys.path.insert(0,str(SOURCE))
import engine
class MA276Tests(unittest.TestCase):
 def test_butterfly_is_orthogonal(self):
  b=engine.boft(.31);self.assertTrue(np.allclose(b@b.T,np.eye(engine.D),atol=1e-12))
 def test_payload_roundtrip_and_mirror_teacher(self):
  rows=engine.run(27611,'fresh',2);a={r['method']:r for r in rows if r['condition']=='aligned'}
  self.assertLess(a['mirror_boft']['layer_output_mse'],1e-12)
  self.assertLess(a['mirror_boft']['composed_output_mse'],1e-12)
  self.assertTrue(all(r['runtime_path_max_abs_diff']<3e-7 for r in rows))
  self.assertGreater(a['mirror_boft']['inference_examples_per_s'],1e6)
  self.assertEqual(a['mirror_boft']['operator_workspace_bytes'],4*engine.D*engine.D*8)
 def test_full_upper_control_recovers_independent_layers(self):
  rows=engine.run(27611,'fresh',2);a={r['method']:r for r in rows if r['condition']=='independent'}
  self.assertLess(a['untied_full']['composed_output_mse'],1e-12)
  self.assertLess(a['boft_per_depth']['serialized_bytes'],a['untied_full']['serialized_bytes'])
 def test_record_codec_is_byte_exact(self):
  p=engine.pack([('method','x'),('t',np.arange(8,dtype=np.float32))]);self.assertEqual(engine.pack(engine.unpack(p)),p)
 def test_replay(self):
  self.assertEqual(len(engine.deterministic_rows(27611,'fresh',2)),14)
if __name__=='__main__':unittest.main()
