from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run_screen as r

class ColumnViewTests(unittest.TestCase):
    def test_walsh_columns_orthonormal(self):
        np.testing.assert_allclose(r.U.T@r.U,np.eye(16),atol=1e-12)

    def test_typed_payload_roundtrip_is_byte_exact(self):
        records=[('f',np.arange(12,dtype=np.float32).reshape(3,4)),('u',np.array([1,3],dtype=np.uint8))]
        payload=r.serialize(records); decoded=r.deserialize(payload)
        self.assertEqual(payload,r.serialize(list(decoded.items())))
        np.testing.assert_array_equal(decoded['f'],records[0][1]); np.testing.assert_array_equal(decoded['u'],records[1][1])

    def test_integer_address_and_onehot_control_are_functionally_equal(self):
        world,codes=r.build_world(28611)
        pi,_,_=r.build_method('mirror_index',world,codes); po,_,_=r.build_method('mirror_onehot',world,codes)
        ri,ro=r.deserialize(pi),r.deserialize(po)
        for t in range(8):
            x=world[t][2][0]
            np.testing.assert_allclose(r.predict(ri,'mirror_index',t,x),r.predict(ro,'mirror_onehot',t,x),atol=1e-6)

    def test_unrelated_tasks_use_private_fallback(self):
        world,codes=r.build_world(28611); payload,_,private=r.build_method('mirror_index',world,codes)
        rec=r.deserialize(payload); self.assertEqual(private,2)
        self.assertEqual(rec['task_kinds'][-2:].tolist(),[2,2])
        for t in (8,9): self.assertLess(r.nmse(r.predict(rec,'mirror_index',t,world[t][2][0]),world[t][2][1]),1e-5)

    def test_matched_cheap_lora_reconstructs_related_maps(self):
        world,codes=r.build_world(28611); payload,_,_=r.build_method('cheap_matched',world,codes); rec=r.deserialize(payload)
        for t in range(8): self.assertLess(r.nmse(r.predict(rec,'cheap_matched',t,world[t][2][0]),world[t][2][1]),1e-5)

if __name__=='__main__': unittest.main()
