import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
import engine

class ProgramMemoryTests(unittest.TestCase):
    def test_corner_memory_interpolates_affine_program_exactly(self):
        a0,b1,b2,codes,private=engine.world(65501)
        meta,arr=engine.state('nspm_corner_memory',65501)
        for z in ([-.7,.2],[.3,-.8],[0.0,0.0]):
            got=engine.controller(meta['method'],meta,arr,z,0)
            want=engine.affine(a0,b1,b2,z)
            self.assertLess(float(abs(got-want).max()),1e-5)
    def test_mirror_and_plain_basis_have_same_quality_and_bytes(self):
        m=engine.eval_one(65501,'mirror_shared_program_code')
        b=engine.eval_one(65501,'ordinary_shared_matrix_basis')
        self.assertAlmostEqual(m['normalized_trajectory_mse'],b['normalized_trajectory_mse'],places=12)
        self.assertLess(abs(m['serialized_bytes']-b['serialized_bytes']),16)
    def test_private_program_is_serialized(self):
        meta,arr=engine.state('mirror_shared_program_code',65501)
        self.assertEqual(arr['private'].shape,(engine.D,engine.D))
        blob=engine.serialize(meta,arr);meta2,arr2=engine.deserialize(blob)
        self.assertEqual(engine.serialize(meta2,arr2),blob)

if __name__=='__main__':unittest.main()
