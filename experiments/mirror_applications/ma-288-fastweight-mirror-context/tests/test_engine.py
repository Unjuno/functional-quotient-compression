import sys,unittest
from pathlib import Path
import numpy as np
SOURCE=Path(__file__).resolve().parents[1]/'source';sys.path.insert(0,str(SOURCE))
import engine
class MA288Tests(unittest.TestCase):
 def test_record_codec_roundtrip_is_byte_exact(self):
  payload=engine.pack([('method','mirror'),('state',np.arange(12,dtype=np.float32).reshape(3,4))])
  self.assertEqual(engine.pack(engine.unpack(payload)),payload)
 def test_fast_weight_delta_write_recovers_basis_memory(self):
  w=np.arange(64,dtype=float).reshape(8,8)/10
  self.assertTrue(np.allclose(engine.learn_fwp(w),w))
 def test_aligned_mirror_and_independent_upper(self):
  rows=engine.run(28811,'fresh',2);aligned={r['method']:r for r in rows if r['condition']=='aligned'};ind={r['method']:r for r in rows if r['condition']=='independent'}
  self.assertLess(aligned['mirror']['task_output_mse'],1e-12)
  self.assertLess(ind['fwp_independent']['task_output_mse'],1e-12)
  self.assertGreater(ind['mirror']['task_output_mse'],1e-4)
 def test_mirror_state_is_cheaper_than_independent_and_replays(self):
  rows=engine.deterministic_rows(28811,'fresh',2)
  for condition in ['aligned','independent']:
   sub={r['method']:r for r in rows if r['condition']==condition}
   self.assertLess(sub['mirror']['serialized_bytes'],sub['fwp_independent']['serialized_bytes'])
 def test_reconstruction(self):
  rows=engine.run(28811,'fresh',2)
  self.assertTrue(all(r['reconstruction_max_abs_diff']<1e-6 for r in rows if r['method']=='fwp_independent'))
  self.assertTrue(all(r['runtime_path_max_abs_diff']<3e-7 for r in rows))
if __name__=='__main__':unittest.main()
