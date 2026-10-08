import sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
import engine

class EngineTests(unittest.TestCase):
    def test_payload_roundtrip_and_exact_bytes(self):
        k,v,a,q=engine.teacher(33031)
        meta,arrays=engine.make_state('mirror_shared_base_private_boundary',4,k,v,a)
        payload=engine.serialize(meta,arrays)
        m2,a2=engine.deserialize(payload)
        self.assertEqual(engine.serialize(m2,a2),payload)
        kh,vh=engine.reconstruct(m2,a2)
        self.assertLess(engine.score(kh,vh,k,v,q),1e-10)
    def test_pa35_private_boundary_reconstructs(self):
        k,v,a,q=engine.teacher(33031)
        meta,arrays=engine.make_state('pa35_shared_sequence_basis',4,k,v,a)
        kh,vh=engine.reconstruct(*engine.deserialize(engine.serialize(meta,arrays)))
        self.assertLess(engine.score(kh,vh,k,v,q),1e-10)
    def test_unrelated_roles_have_private_factors(self):
        k,v,a,q=engine.teacher(33031)
        _,arrays=engine.make_state('mirror_shared_base_private_boundary',4,k,v,a)
        for name in ('k_private_u_06','k_private_u_07','v_private_u_06','v_private_u_07'):
            self.assertGreater(arrays[name].size,0)

if __name__=='__main__': unittest.main()
