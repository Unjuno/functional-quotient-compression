import hashlib,json,unittest
from pathlib import Path
import numpy as np
E=Path(__file__).resolve().parents[1];R=E/'runs';D=json.loads((R/'exploratory_one_object.json').read_text())
class SAM2LongPathMemoryTests(unittest.TestCase):
 def test_protocol_deviation_is_explicit(self):
  p=json.loads((E/'PROTOCOL.json').read_text())
  self.assertEqual(p['scene']['objects'],2)
  self.assertEqual(D['scope'],'Not a valid registered 2-object run: exploratory single-object SAM2Long tree only, one seed, 12 frames.')
 def test_full_forward_predictions_are_identical(self):
  rows=D['results'];self.assertEqual(len(rows),4)
  hashes={r['prediction_sha256'] for r in rows};self.assertEqual(len(hashes),1)
  arrays=[np.load(R/f"pred_seed_{rows[0]['seed']}_{r['method']}.npz")['masks'] for r in rows]
  for a in arrays[1:]:np.testing.assert_array_equal(a,arrays[0])
 def test_metrics_and_payload_arithmetic(self):
  for r in D['results']:
   self.assertAlmostEqual(r['mean_jf'],0.8440573780362298,places=12)
   self.assertEqual(r['full_session_payload_bytes'],D['checkpoint_bytes']+D['config_bytes']+D['common_video_cache']['bytes']+r['session_payload_bytes'])
 def test_retained_tensor_manifests_match_payload_sizes(self):
  headers=json.loads((R/'exploratory_session_headers.json').read_text())
  for row in D['results']:
   h=headers[row['method']];header_bytes=len(json.dumps(h,sort_keys=True,separators=(',',':')).encode())+1
   tensor_bytes=sum(t['bytes'] for t in h['tensors'])
   self.assertEqual(header_bytes+tensor_bytes,row['session_payload_bytes'])
   self.assertEqual(tensor_bytes,sum(t['bytes'] for t in h['tensors']))
 def test_complete_session_storage_gate_is_impossible_for_observed_shape(self):
  b=D['structural_storage_bound']
  self.assertLess(b['largest_possible_full_session_savings_if_path_memory_were_free'],b['required_savings_fraction'])
 def test_official_tree_is_single_object_path_logic(self):
  src=Path('/tmp/ma1028-sam2long/sam2/sam2_video_predictor.py').read_text()
  self.assertIn('object_score.abs().item()',src)
  base=Path('/tmp/ma1028-sam2long/sam2/modeling/sam2_base.py').read_text()
  self.assertIn('always one',base)
if __name__=='__main__':unittest.main()
