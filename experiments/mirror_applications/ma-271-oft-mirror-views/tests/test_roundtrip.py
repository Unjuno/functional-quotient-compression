import io,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run as e
class Verify(unittest.TestCase):
 def test_views_reconstruct_and_are_invertible(self):
  w,q,a,tr,te=e.world(127,True);maps=e.fitmaps(tr);bl=e.fit_block(w,maps);phi,m,rankpred=e.fit_rank1(w,maps,bl)
  cases=[('oft_dense',[w,q],np.array([x@w for x in q])),('oft_blockwise',[w,bl],np.array([e.qmat(x)@w for x in bl])),('simple_rank1_angle',[w,np.r_[phi,m]],rankpred),('mirror_view',[w,np.r_[phi,m]],rankpred)]
  for name,state,pred in cases:
   b=io.BytesIO();np.savez(b,**{f'p{i}':x for i,x in enumerate(state)});b.seek(0)
   with np.load(b) as z:s=[z[k] for k in z.files]
   if name=='oft_dense':out=np.array([x@s[0] for x in s[1]])
   elif name=='oft_blockwise':out=np.array([e.qmat(x)@s[0] for x in s[1]])
   else:
    ph=s[1][:e.P];code=s[1][e.P:];out=np.array([e.qmat(code[i]*ph)@s[0] for i in range(e.T)])
   self.assertTrue(np.array_equal(pred,out))
   self.assertLess(e.ortho(np.array([np.linalg.qr(np.eye(e.D))[0]])),1e-12)
if __name__=='__main__':unittest.main()
