import sys,unittest
from pathlib import Path
import numpy as np
SOURCE=Path(__file__).resolve().parents[1]/'source';sys.path.insert(0,str(SOURCE))
import engine
class MA292Tests(unittest.TestCase):
 def test_payload_roundtrip_and_serialized_reconstruction(self):
  rows=engine.run(29211,'fresh',4)
  self.assertEqual(len(rows),12)
  self.assertTrue(all(r['serialized_bytes']>0 for r in rows))
  self.assertTrue(all(r['optimizer_updates']==0 for r in rows))
  full=[r for r in rows if r['method']=='independent_full']
  self.assertTrue(all(r['max_delta_abs_error']<1e-6 for r in full))
 def test_record_codec_is_byte_exact(self):
  records=[('method','mirror_angle'),('tensor',np.arange(8,dtype=np.float32).reshape(2,4)),('half',np.array([.25,-.5],dtype=np.float16))]
  payload=engine._payload_pack(records)
  self.assertEqual(engine._payload_pack(engine._payload_unpack(payload)),payload)
 def test_mirror_has_smaller_code_and_svd_has_zeroish_projection_error(self):
  rows=engine.run(29211,'fresh',4)
  a={r['method']:r for r in rows if r['condition']=='aligned'}
  self.assertLess(a['mirror_angle']['incremental_task_bytes'],a['svd_rank2']['incremental_task_bytes'])
  self.assertEqual(a['mirror_angle']['incremental_task_bytes'],a['svd_rank2_fp16']['incremental_task_bytes'])
  self.assertLess(a['svd_rank2']['task_output_mse'],a['svd_rank1']['task_output_mse'])
 def test_independent_tasks_need_private_state(self):
  rows=engine.run(29211,'fresh',4);a={r['method']:r for r in rows if r['condition']=='independent'}
  self.assertGreater(a['mirror_angle']['task_output_mse'],a['independent_full']['task_output_mse']*1e6)
 def test_replay(self):
  rows=engine.deterministic_rows(29211,'fresh',4);self.assertEqual(len(rows),12)
if __name__=='__main__':unittest.main()
