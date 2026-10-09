import csv,hashlib
from pathlib import Path
import numpy as np
import run
ROOT=Path(__file__).resolve().parents[1];rows=list(csv.DictReader((ROOT/'FRESH_RESULTS.csv').open()));maxdiff=0.
for seed in run.FRESH:
 a=run.world(seed);desc=np.stack([np.cos(a['phte']),np.sin(a['phte'])],-1).astype(np.float32);phase=a['phte'][:,None].astype(np.float32);zero=np.zeros((8,0),np.float32)
 results={'mirror_phase':run.basis(a,True),'generic_coefficients':run.basis(a,False),'hyperlora_generator':run.hyper(a),'independent_client_lora':run.independent(a),'base_only':run.shared(a)}
 for method,(mse,state,sec,trainloss) in results.items():
  codes=phase if method=='mirror_phase' else desc if method in ('generic_coefficients','hyperlora_generator') else zero
  blob=run.pack(method,a['base'],state,codes);r=next(x for x in rows if int(x['world'])==seed and x['method']==method)
  assert len(blob)==int(r['serialized_bytes']) and hashlib.sha256(blob).hexdigest()==r['payload_sha256']
  maxdiff=max(maxdiff,abs(mse-float(r['heldout_test_mse'])))
assert maxdiff<1e-10,maxdiff
print({'rows':len(rows),'metric_replay_max_abs_difference':maxdiff,'payload_hashes':'exact'})
