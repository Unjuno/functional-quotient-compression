import io,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run as e
class Verify(unittest.TestCase):
 def test_packed_skews_reconstruct_and_neumann_converges(self):
  w,ss,amps,Q,xs,ys=e.world(149,'shared_generator_orbit')
  packed=e.pack_skew(ss*amps[:,None,None]);rest=np.array([e.unpack_skew(v) for v in packed]);self.assertTrue(np.allclose(ss*amps[:,None,None],rest))
  for k in range(e.T):
   q=e.cayley(rest[k]);self.assertTrue(np.allclose(q,Q[k],atol=1e-14))
   self.assertLess(np.linalg.norm(q.T@q-np.eye(e.D)),1e-12)
  a=amps[0]*ss[0];q8=e.neumann(a,8);q=e.cayley(a);self.assertLess(np.linalg.norm(q8-q),1e-7)
  buf=io.BytesIO();np.savez(buf,W=w,S=e.pack_skew(ss),m=amps);self.assertGreater(len(buf.getvalue()),0)
if __name__=='__main__':unittest.main()
