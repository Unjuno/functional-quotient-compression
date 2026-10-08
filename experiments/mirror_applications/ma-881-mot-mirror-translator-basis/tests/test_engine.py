import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
import engine

class TranslatorViewTests(unittest.TestCase):
    def test_givens_view_matches_teacher_experts(self):
        w0,angles,weights,router,corr=engine.world(88101)
        for i in range(engine.ALIGNED):
            self.assertLess(float(abs(engine.rotmat(angles[i])@w0-weights[i]).max()),1e-12)
    def test_mirror_trajectory_reconstructs_and_payload_roundtrips(self):
        meta,a=engine.state('mirror_shared_givens_views',0,88101)
        blob=engine.serialize(meta,a);m2,a2=engine.deserialize(blob)
        self.assertEqual(engine.serialize(m2,a2),blob)
        x,ctx=engine.make_eval(88101,2);w0,angles,weights,r,c=engine.world(88101)
        ref={'weights':weights.astype('<f4'),'router':r.astype('<f4'),'corr':c.astype('<f4')}
        for xx,cc in zip(x,ctx):
            yref,_=engine.trajectory('native_mot_full',{},ref,xx,cc)
            yhat,_=engine.trajectory(m2['method'],m2,a2,xx,cc)
            self.assertLess(float(((yhat-yref)**2).mean()),1e-10)
    def test_private_expert_is_paid(self):
        _,a=engine.state('mirror_shared_givens_views',0,88101)
        self.assertEqual(a['private'].shape,(engine.D,engine.D))

if __name__=='__main__':unittest.main()
