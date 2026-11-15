import sys,unittest
from pathlib import Path
E=Path(__file__).resolve().parents[1];sys.path.insert(0,str(E/'source'))
from verify_result import verify
class MA1030ResultTests(unittest.TestCase):
    def test_saved_predictions_payloads_and_metrics_replay(self):self.assertTrue(verify()['passed'])
if __name__=='__main__':unittest.main()
