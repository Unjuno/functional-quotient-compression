"""Post-audit diagnostic: amortize a dynamic session view into a temporary cache."""
import csv,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import engine
ROOT=HERE.parent
FIELDS=['seed','condition','method','precompute_s','active_session_workspace_bytes','all_context_workspace_bytes','queries_per_context','amortized_examples_per_s']
rows=[]
for seed in (28811,28812,28813):
 for condition in ('aligned','independent'):
  base,angles,teachers,queries=engine.make_world(seed,condition)
  base_state=engine.learn_fwp(teachers[0])
  for method in ('mirror','lowrank_residual','fwp_independent'):
   payload=engine.serialize(method,base_state,angles,teachers,4)
   t0=time.perf_counter();cached=engine.restore(payload,4);pre=time.perf_counter()-t0
   for nquery in (1,8,64,512):
    t0=time.perf_counter()
    for _ in range(nquery):
     for c in range(engine.CONTEXTS): _=queries[c]@cached[c]
    elapsed=time.perf_counter()-t0
    rows.append(dict(seed=seed,condition=condition,method=method,precompute_s=pre,active_session_workspace_bytes=engine.D*engine.D*4,all_context_workspace_bytes=engine.CONTEXTS*engine.D*engine.D*4,queries_per_context=nquery,amortized_examples_per_s=nquery*engine.CONTEXTS*engine.N_QUERY/max(pre+elapsed,1e-12)))
with (ROOT/'CACHE_RUNTIME_DIAGNOSTIC.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
print(ROOT/'CACHE_RUNTIME_DIAGNOSTIC.csv')
