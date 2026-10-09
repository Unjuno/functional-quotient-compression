import csv,hashlib
from pathlib import Path
import run
ROOT=Path(__file__).resolve().parents[1];rows=list(csv.DictReader((ROOT/'FRESH_RESULTS.csv').open()));maxdiff=0.
for seed in run.FRESH:
 a=run.world(seed);mods={'map':run.fit_map(a),'mirror':run.fit_mirror(a),'rank1_bnn':run.fit_rank1(a)}
 for name,(state,sec) in mods.items():
  blob=run.pack(name,state);h=hashlib.sha256(blob).hexdigest()
  for split in ('iid','ood'):
   vals=run.metrics(name,state,a,split);r=next(x for x in rows if int(x['world'])==seed and x['method']==name and x['split']==split)
   assert len(blob)==int(r['serialized_bytes']) and h==r['payload_sha256']
   for key,val in zip(('nll','brier','ece_10','accuracy'),vals[:4]):maxdiff=max(maxdiff,abs(val-float(r[key])))
assert maxdiff<1e-12,maxdiff
print({'rows':len(rows),'metric_replay_max_abs_difference':maxdiff,'payload_hashes':'exact'})
