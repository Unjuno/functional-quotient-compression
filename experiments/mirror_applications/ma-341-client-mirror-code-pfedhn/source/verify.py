import csv,hashlib
from pathlib import Path
import numpy as np
import run
ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((ROOT/'FRESH_RESULTS.csv').open()));maxdiff=0.
for seed in run.FRESH:
 a=run.world(seed);desc=np.stack([np.cos(a['ph_test']),np.sin(a['ph_test'])],-1).astype(np.float32);phase=a['ph_test'][:,None].astype(np.float32)
 calc={'shared_single':run.fit_shared(a),'mirror_phase':run.fit_basis(a,'mirror'),'generic_coefficients':run.fit_basis(a,'generic'),'pfedhn_generator':run.fit_hyper(a),'independent_client_fit':run.fit_independent(a)}
 for method,(mse,state,sec,trainloss) in calc.items():
  codes=phase if method=='mirror_phase' else desc if method in ('shared_single','generic_coefficients','pfedhn_generator') else np.zeros((8,0),np.float32)
  blob=run.pack(method,state,[codes]);r=next(x for x in rows if int(x['world'])==seed and x['method']==method)
  assert len(blob)==int(r['serialized_bytes']) and hashlib.sha256(blob).hexdigest()==r['payload_sha256']
  maxdiff=max(maxdiff,abs(mse-float(r['heldout_test_mse'])))
assert maxdiff<1e-10,maxdiff
print({'rows':len(rows),'metric_replay_max_abs_difference':maxdiff,'payload_hashes':'exact'})
