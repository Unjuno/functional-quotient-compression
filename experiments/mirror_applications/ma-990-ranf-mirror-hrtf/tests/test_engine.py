import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import engine

class HRTFScreenTests(unittest.TestCase):
    def test_binary_payload_round_trip_and_byte_exact(self):
        x=np.arange(24,dtype=np.float32).reshape(3,2,4)
        blob=engine.encode({'kind':'test'},{'map':x,'indices':np.array([1,3],dtype=np.uint8)})
        meta,arr=engine.decode(blob)
        self.assertEqual(meta['kind'],'test')
        np.testing.assert_array_equal(arr['map'],x)
        np.testing.assert_array_equal(arr['indices'],[1,3])
        self.assertEqual(blob,engine.encode({'kind':'test'},{'map':x,'indices':np.array([1,3],dtype=np.uint8)}))

    def test_mirror_sparse_code_is_ordinary_sparse_control(self):
        mean=np.zeros((engine.DIRECTIONS,engine.EARS,engine.SAMPLES),dtype=np.float32)
        basis=np.zeros((engine.RANK,engine.DIRECTIONS,engine.EARS,engine.SAMPLES),dtype=np.float32)
        target=np.random.default_rng(2).normal(size=mean.shape).astype(np.float32)
        idx,coef=engine.fit_omp(mean,basis,target,np.array([4,11,414]))
        mirror=engine.personalization_payload(coef,idx,listener_id=37)
        ordinary=engine.personalization_payload(coef,idx,listener_id=37)
        self.assertEqual(mirror,ordinary)
        other_listener=engine.personalization_payload(coef,idx,listener_id=38)
        self.assertNotEqual(mirror,other_listener)

    def test_predict_code_shape_and_support_constants(self):
        mean=np.zeros((engine.DIRECTIONS,engine.EARS,engine.SAMPLES),dtype=np.float32)
        basis=np.ones((engine.RANK,engine.DIRECTIONS,engine.EARS,engine.SAMPLES),dtype=np.float32)
        pred=engine.predict_code(mean,basis,np.ones(4,dtype=np.float32),np.array([0,2,4,6]))
        self.assertEqual(pred.shape,(793,2,256))
        self.assertEqual(len(engine.SUPPORTS[3]),3)
        self.assertEqual(len(engine.SUPPORTS[5]),5)

if __name__=='__main__':unittest.main()
