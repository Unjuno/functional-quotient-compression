import csv,unittest
from pathlib import Path
R=Path(__file__).resolve().parents[1]
class Audit(unittest.TestCase):
 def test_fresh_not_opened(self):
  import json
  self.assertFalse(json.loads((R/"PROTOCOL.json").read_text())["metric_audit_after_development"]["fresh_opened"])
 def test_development_rows_exist(self):
  self.assertGreater(len(list(csv.DictReader((R/"DEVELOPMENT_RESULTS.csv").open()))),0)
if __name__=="__main__":unittest.main()
