import io,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run as e
class RoundTrip(unittest.TestCase):
 def test_payload_reconstructs_each_expert_bank(self):
  _,tr,te=e.data(107,'aligned_givens');w=e.fit(tr);shared=w.mean(0);be,bes,_=e.befit(w);mi,mis,_=e.mirfit(w)
  cases=[('independent',w,[w]),('hard_tying',np.tile(shared,(e.N,1,1)),[shared]),('batchensemble_rank1',be,bes),('mirror_givens',mi,mis)]
  for name,pred,state in cases:
   b=io.BytesIO();np.savez(b,**{f'p{i}':np.asarray(v) for i,v in enumerate(state)});b.seek(0)
   with np.load(b) as z: loaded=[z[k] for k in z.files]
   if name=='independent':decoded=loaded[0]
   elif name=='hard_tying':decoded=np.tile(loaded[0],(e.N,1,1))
   elif name=='batchensemble_rank1':decoded=np.array([loaded[0]*np.outer(loaded[1][k],loaded[2][k]) for k in range(e.N)])
   else:decoded=np.array([e.qmat(a)@loaded[0] for a in loaded[1]])
   self.assertTrue(np.array_equal(pred,decoded),name)
   self.assertGreater(e.nbytes(state),0)
   for p,(x,y) in zip(decoded,te):self.assertTrue(np.isfinite(np.mean((x@p-y)**2)))
if __name__=='__main__':unittest.main()
