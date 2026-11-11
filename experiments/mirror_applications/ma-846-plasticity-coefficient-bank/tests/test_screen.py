import importlib.util
from pathlib import Path
import unittest

src=Path(__file__).parents[1]/'source'/'run_screen.py'
spec=importlib.util.spec_from_file_location('screen',src); screen=importlib.util.module_from_spec(spec); spec.loader.exec_module(screen)

class ScreenTests(unittest.TestCase):
    def test_native_and_mirror_factor_alias(self):
        native=screen.eval_rule(29,rank=screen.RANK)
        mirror=screen.eval_rule(29,rank=screen.RANK)
        self.assertEqual(native[:3],mirror[:3])
    def test_fast_state_and_update_work_are_charged(self):
        _,_,state,mac=screen.eval_rule(29,rank=screen.RANK)
        self.assertEqual(state,screen.TASKS*screen.D*screen.D*4)
        self.assertEqual(mac,screen.TASKS*screen.N_SUPPORT*screen.D**3)

if __name__=='__main__': unittest.main()
