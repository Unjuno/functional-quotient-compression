import csv,unittest
from pathlib import Path
R=Path(__file__).resolve().parents[1]
class Audit(unittest.TestCase):
 def test_fresh_not_opened(self):
  import json
  p=json.loads((R/"PROTOCOL.json").read_text())
  self.assertFalse(p["fresh"]["locked_before_access"] is False)
  self.assertFalse(p["amendment_A1"]["fresh_opened_at_freeze"])
 def test_development_rows_exist(self):
  self.assertGreater(len(list(csv.DictReader((R/"DEVELOPMENT_RESULTS_A1.csv").open()))),0)
 def test_taskwise_metric_is_finite_and_control_alias_is_explicit(self):
  rows=list(csv.DictReader((R/"DEVELOPMENT_RESULTS_A1.csv").open()))
  self.assertTrue(all(float(x["nrmse"]) < 2 for x in rows))
  m={(x["world"],x["seed"],x["task"]):x for x in rows if x["method"].startswith("mirror_r8_lr0.01")}
  c={(x["world"],x["seed"],x["task"]):x for x in rows if x["method"].startswith("generic_lowrank_r8_lr0.01")}
  self.assertEqual(len(m),len(c));self.assertTrue(all(m[k]["nrmse"]==c[k]["nrmse"] for k in m))
if __name__=="__main__":unittest.main()
