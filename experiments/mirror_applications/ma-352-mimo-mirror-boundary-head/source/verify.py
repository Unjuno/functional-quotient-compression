import csv,hashlib
from pathlib import Path
import run
ROOT=Path(__file__).resolve().parents[1];rows=list(csv.DictReader((ROOT/'FRESH_RESULTS.csv').open()));md=0.
for seed in run.FRESH:
 a=run.world(seed)
 for method in ('mimo','mirror','generic','shared'):
  state,_,_=run.train(a,method);w=run.build(method,state);blob=run.pack(method,a,state);vals=run.metrics(method,a,w);r=next(x for x in rows if int(x['world'])==seed and x['method']==method)
  assert len(blob)==int(r['serialized_bytes']) and hashlib.sha256(blob).hexdigest()==r['payload_sha256']
  for x,y in zip(vals,(float(r['nll']),float(r['brier']),float(r['ece_10']),float(r['accuracy']),float(r['member_logit_correlation']),float(r['pairwise_disagreement']))):md=max(md,abs(x-y))
assert md<1e-12,md
print({'rows':len(rows),'metric_replay_max_abs_difference':md,'payload_hashes':'exact'})
