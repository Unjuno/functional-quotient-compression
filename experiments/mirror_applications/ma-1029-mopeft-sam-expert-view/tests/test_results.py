import sys,unittest
from pathlib import Path
E=Path(__file__).resolve().parents[1];sys.path.insert(0,str(E/'source'))
from verify_result import verify

class SavedAuditVerificationTests(unittest.TestCase):
    def test_saved_audit_predictions_metrics_payloads_and_bytes_replay(self):
        self.assertTrue(verify()['passed'])

if __name__=='__main__':unittest.main()
