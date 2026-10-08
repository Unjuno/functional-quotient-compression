import io,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run as e
class Verify(unittest.TestCase):
 def test_boft_is_orthogonal_and_shared_factor_state_replays(self):
  w,Q,a,tr,te=e.world(137,'aligned_angle_orbit');maps=e.fitmaps(tr);ang=e.fit_angles(w,maps);phi,m,fac=e.fit_factor(w,maps,ang)
  for v in list(ang)+[v*phi for v in m]:self.assertLess(np.linalg.norm(e.boft(v).T@e.boft(v)-np.eye(e.D)),1e-12)
  state=[w,np.r_[phi,m]]
  buf=io.BytesIO();np.savez(buf,**{f'p{i}':x for i,x in enumerate(state)});buf.seek(0)
  with np.load(buf) as z:s=[z[k] for k in z.files]
  phi2=s[1][:e.K];m2=s[1][e.K:];out=np.array([e.boft(v*phi2)@s[0] for v in m2])
  self.assertTrue(np.allclose(out,fac,atol=1e-14,rtol=0))
if __name__=='__main__':unittest.main()
