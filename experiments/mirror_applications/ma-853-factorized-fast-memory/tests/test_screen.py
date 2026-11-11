import importlib.util
from pathlib import Path
import unittest
p=Path(__file__).parents[1]/'source'/'run.py';sp=importlib.util.spec_from_file_location('r',p);r=importlib.util.module_from_spec(sp);sp.loader.exec_module(r)
class TestRoleComposition(unittest.TestCase):
 def test_composition_and_controls(self):
  x=r.run(61)['rows']
  self.assertEqual(x['native']['nrmse'],x['mirror']['nrmse'])
  self.assertEqual(x['native']['bytes'],x['mirror']['bytes'])
  self.assertEqual(x['native']['query_macs'],x['mirror']['query_macs'])
  self.assertLess(x['native']['bytes'],x['full']['bytes']*.1)
  self.assertEqual(x['hadamard']['nrmse'],0.)
if __name__=='__main__':unittest.main()
