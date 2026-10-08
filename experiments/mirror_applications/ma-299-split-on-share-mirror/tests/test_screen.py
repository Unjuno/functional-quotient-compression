from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run_screen as r

class SplitOnShareTests(unittest.TestCase):
    def test_view_rotation_is_orthogonal(self):
        for angle in (-1.1,-.2,0.,.7):
            np.testing.assert_allclose(r.rotation(angle)@r.rotation(angle).T,np.eye(r.D),atol=1e-12)

    def test_binary_serializer_roundtrip(self):
        records=[('float',np.arange(8,dtype=np.float32)),('byte',np.array([1,2,3],dtype=np.uint8))]
        payload=r.serialize(records); got=r.deserialize(payload)
        np.testing.assert_array_equal(got['float'],records[0][1])
        np.testing.assert_array_equal(got['byte'],records[1][1])
        self.assertEqual(payload,r.serialize(records))

    def test_one_angle_matches_two_coefficient_basis(self):
        rng=np.random.default_rng(9); base=rng.normal(size=(r.D,r.D)); x=rng.normal(size=(128,r.D)); theta=.37
        y=(x@base)@r.rotation(theta); coeff=np.array([np.cos(theta),np.sin(theta)])
        y0=x@base; control=coeff[0]*y0+coeff[1]*(y0@r.G)
        np.testing.assert_allclose(y,control,atol=1e-12)

    def test_unrelated_task_triggers_private_split(self):
        world=r.build_world(29911); base=r.fit_full(*world[0][0]); xtr,ytr=world[4][0]; xv,yv=world[4][1]
        s=r.candidate_for('mirror_split',4,xtr,ytr,xv,yv,base,.01)
        self.assertEqual(s['kind'],'private')

    def test_payload_execution_uses_serialized_task_state(self):
        world=r.build_world(29911); base=r.fit_full(*world[0][0]); states=[{'kind':'share'}]
        angle=.4; states.append({'kind':'view','theta':angle})
        payload=r.serialize_states('mirror_split',base,states); loaded=r.deserialize(payload)
        x=world[1][2][0]
        np.testing.assert_allclose(r.predict_records(loaded,1,x),(x@base)@r.rotation(angle),atol=2e-6)

if __name__=='__main__': unittest.main()
