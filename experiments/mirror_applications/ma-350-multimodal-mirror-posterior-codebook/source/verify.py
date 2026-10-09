import csv,hashlib
from pathlib import Path
import run
ROOT=Path(__file__).resolve().parents[1];rows=list(csv.DictReader((ROOT/'FRESH_RESULTS.csv').open()));md=0.
for seed in run.FRESH:
 a=run.world(seed)
 for method in run.METHODS:
  blob=run.pack(method,a);meta,arr=run.load(blob)
  for split in ('iid','ood'):
   vals=run.predictions(method,a,split);r=next(x for x in rows if int(x['world'])==seed and x['method']==method and x['split']==split)
   assert len(blob)==int(r['serialized_bytes']) and hashlib.sha256(blob).hexdigest()==r['payload_sha256']
   for key,val in zip(('nll','brier','ece_10','accuracy','pairwise_disagreement'),vals):md=max(md,abs(val-float(r[key])))
assert md<1e-12,md
print({'rows':len(rows),'metric_replay_max_abs_difference':md,'payload_hashes':'exact'})
