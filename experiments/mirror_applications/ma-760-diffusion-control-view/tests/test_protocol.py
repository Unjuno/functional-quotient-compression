import json, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class ProtocolTests(unittest.TestCase):
 def test_draw_matches_protocol_and_pool(self):
  d=json.loads((ROOT/"source/random_draw.json").read_text());p=json.loads((ROOT/"PROTOCOL.json").read_text())
  self.assertEqual(d["selected_id"],"MA-760");self.assertEqual(p["experiment_id"],d["selected_id"])
  self.assertEqual(d["pool_size"],529);self.assertEqual(d["zero_based_index"],259)
 def test_fresh_is_locked(self):
  p=json.loads((ROOT/"PROTOCOL.json").read_text())
  self.assertTrue(p["fresh"]["locked_before_access"]);self.assertTrue(p["gates"]["fresh_opening"].startswith("Only if"))
 def test_total_payload_counts_base_and_private(self):
  p=json.loads((ROOT/"PROTOCOL.json").read_text())
  paid=p["storage"]["paid_objects"]
  self.assertTrue(any("UNet" in x for x in paid));self.assertTrue(any("private" in x for x in paid))
if __name__=="__main__":unittest.main()
