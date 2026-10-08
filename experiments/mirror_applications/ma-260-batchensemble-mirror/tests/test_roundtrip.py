import io,json,subprocess,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run as exp
class Verify(unittest.TestCase):
 def test_actual_fresh_states_and_metrics_are_finite(self):
  rows=exp.run(103,'rotation-aligned')
  for row in rows:
   self.assertGreater(row['serialized_bytes'],0)
   for k in ('accuracy','ece','member_disagreement'):
    self.assertTrue(np.isfinite(row[k]))
  w,tr,te=exp.make_world(103,'rotation-aligned')
  fit=[exp.fit_logistic(x,y)[:2] for x,y in tr]
  models=[('independent',fit,[np.array([a for a,b in fit]),np.array([b for a,b in fit])]),('batchensemble',[(np.array([a for a,b in fit]).mean(0)*r,b) for r,(_,b) in zip(np.array([a for a,b in fit])/(np.array([a for a,b in fit]).mean(0)[None,:]+1e-12),fit)],[np.array([a for a,b in fit]).mean(0),np.array([a for a,b in fit])/(np.array([a for a,b in fit]).mean(0)[None,:]+1e-12),np.array([b for a,b in fit])])]
  for name,model,state in models:
   buf=io.BytesIO();np.savez(buf,**{f'x{i}':np.asarray(v) for i,v in enumerate(state)});buf.seek(0)
   with np.load(buf) as z:loaded=[z[k] for k in z.files]
   self.assertTrue(all(np.array_equal(a,b) for a,b in zip(state,loaded)))
if __name__=='__main__':unittest.main()
