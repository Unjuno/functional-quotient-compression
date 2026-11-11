import importlib.util
from pathlib import Path
import unittest
p=Path(__file__).parents[1]/'source'/'run.py';sp=importlib.util.spec_from_file_location('r',p);r=importlib.util.module_from_spec(sp);sp.loader.exec_module(r)
class TestRuleViews(unittest.TestCase):
 def test_native_gate_alias_and_reset(self):
  x=r.run(43,.45)['results']
  self.assertEqual(x['native_gate']['nrmse'],x['mirror_code']['nrmse'])
  self.assertEqual(x['native_gate']['bytes'],x['mirror_code']['bytes'])
  self.assertEqual(x['native_gate']['reset_max'],0)
  self.assertEqual(x['mirror_code']['reset_max'],0)
if __name__=='__main__':unittest.main()
