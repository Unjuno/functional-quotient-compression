import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
import engine

class PacketFunctionVectorTests(unittest.TestCase):
    def test_payload_roundtrip_and_ptp_fv_equivalence(self):
        ptp=engine.metrics(53911,'ptp_shared_packet_latent',2,0.0)
        fv=engine.metrics(53911,'mirror_function_vector',2,0.0)
        self.assertLess(abs(ptp['normalized_packet_mse']-fv['normalized_packet_mse']),1e-12)
        self.assertAlmostEqual(ptp['exact_sign_packet_accuracy'],fv['exact_sign_packet_accuracy'])
        self.assertGreaterEqual(fv['serialized_bytes'],ptp['serialized_bytes'])
    def test_private_tasks_are_accounted(self):
        meta,a=engine.build('mirror_function_vector',2,0.0,53911)
        self.assertIn('private_06',a);self.assertIn('private_07',a)
        blob=engine.serialize(meta,a);_,b=engine.deserialize(blob)
        self.assertEqual(b['private_06'].shape,(engine.P,engine.D))
    def test_teacher_shared_code_fits(self):
        row=engine.metrics(53911,'mirror_function_vector',2,0.0)
        self.assertLess(row['normalized_packet_mse'],1e-6)
        self.assertGreater(row['exact_sign_packet_accuracy'],0.99)

if __name__=='__main__':unittest.main()
