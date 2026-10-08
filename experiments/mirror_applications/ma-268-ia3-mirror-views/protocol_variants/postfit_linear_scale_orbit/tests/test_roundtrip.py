import io,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run as e
class RoundTrip(unittest.TestCase):
 def test_payload_states_reconstruct_outputs(self):
  W,tr,te=e.world(111,'aligned_pair_rotation');m=e.fitm(tr);sc=e.gains(m,W);mc,code,_=e.mirfit(sc);mean=sc.mean(0)
  cases=[('independent',m,[m]),('ia3',np.array([np.diag(s)@W for s in sc]),[W,sc]),('shared',np.tile(np.diag(mean)@W,(e.T,1,1)),[W,mean]),('mirror_rotation',np.array([np.diag(s)@W for s in mc]),[W,code])]
  for name,pred,state in cases:
   b=io.BytesIO();np.savez(b,**{f'p{i}':np.asarray(v) for i,v in enumerate(state)});b.seek(0)
   with np.load(b) as z:s=[z[k] for k in z.files]
   if name=='independent':out=s[0]
   elif name=='ia3':out=np.array([np.diag(v)@s[0] for v in s[1]])
   elif name=='shared':out=np.tile(np.diag(s[1])@s[0],(e.T,1,1))
   else:
    b0=s[1][:e.H];angs=s[1][e.H:].reshape(e.T,e.H//2);out=np.array([np.diag(e.view(a)@b0)@s[0] for a in angs])
   self.assertTrue(np.array_equal(pred,out),name);self.assertEqual(e.payload(state),len(b.getvalue()))
if __name__=='__main__':unittest.main()
