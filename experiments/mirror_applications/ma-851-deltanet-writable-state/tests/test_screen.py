import importlib.util
from pathlib import Path
import unittest
p=Path(__file__).parents[1]/'source'/'run.py'; spec=importlib.util.spec_from_file_location('runner',p); runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
class TestDeltaState(unittest.TestCase):
 def test_recall_storage_and_compute(self):
  r=runner.run(41)
  self.assertLess(r['full_nrmse'],1e-6)
  self.assertLess(r['native_lowrank_nrmse'],1e-6)
  self.assertEqual(r['native_lowrank_nrmse'],r['mirror_nrmse'])
  self.assertLess(r['mirror_bytes'],r['full_bytes']*.75)
  self.assertLess(r['lowrank_update_macs'],r['full_update_macs'])
  self.assertLess(r['lowrank_query_macs'],r['full_query_macs'])
if __name__=='__main__': unittest.main()
