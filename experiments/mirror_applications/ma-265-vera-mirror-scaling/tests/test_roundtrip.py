import io,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run as e
class RoundTrip(unittest.TestCase):
 def test_all_payloads_reconstruct(self):
  base,B,A,c,tr,te=e.world(109,'aligned_scale_orbit');m=e.fit_mats(tr);codes=e.fit_codes(m,base,B,A);mir,ms,_=e.mirror_codes(codes);mean=codes.mean(0);lv,rv=e.lowrank_delta(m,base)
  cases=[('independent_dense',m,[m]),('vera',np.array([base+B@np.diag(v)@A for v in codes]),[base,B,A,codes]),('hard_tied',np.tile(base+B@np.diag(mean)@A,(e.T,1,1)),[base,B,A,mean]),('mirror_view',np.array([base+B@np.diag(v)@A for v in mir]),[base,B,A,np.r_[ms[0],ms[1]]]),('lora_rank4',np.array([base+lv[i]@rv[i] for i in range(e.T)]),[base,lv,rv])]
  for name,pred,state in cases:
   b=io.BytesIO();np.savez(b,**{f'p{i}':np.asarray(v) for i,v in enumerate(state)});b.seek(0)
   with np.load(b) as z:s=[z[k] for k in z.files]
   if name=='independent_dense':out=s[0]
   elif name=='vera':out=np.array([s[0]+s[1]@np.diag(v)@s[2] for v in s[3]])
   elif name=='hard_tied':out=np.tile(s[0]+s[1]@np.diag(s[3])@s[2],(e.T,1,1))
   elif name=='mirror_view':
    basev=s[3][:e.R];ang=s[3][e.R:];cc=[]
    for v in ang:
     z=basev.copy();c0,s0=np.cos(v),np.sin(v);z[:2]=[c0*basev[0]-s0*basev[1],s0*basev[0]+c0*basev[1]];cc.append(z)
    out=np.array([s[0]+s[1]@np.diag(v)@s[2] for v in cc])
   else:out=np.array([s[0]+s[1][i]@s[2][i] for i in range(e.T)])
   self.assertTrue(np.array_equal(pred,out),name)
   self.assertEqual(e.payload(state),len(b.getvalue()))
if __name__=='__main__':unittest.main()
