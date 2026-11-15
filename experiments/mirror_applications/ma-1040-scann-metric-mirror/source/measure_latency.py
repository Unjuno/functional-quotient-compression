#!/usr/bin/env python3
"""Five randomized repeated single-query CPU latency blocks after frozen audit."""
import csv,hashlib,json,random,time,os
from pathlib import Path
import numpy as np
import scann
from run_experiment import HERE,DATA,givens,linear

split=json.loads((HERE/'runs/SPLIT_LOCK.json').read_text())
freeze=json.loads((HERE/'runs/DEV_FREEZE.json').read_text())
ids=json.loads((HERE/'runs/ids.json').read_text())
qids=sorted(split['audit_qids']); qindex={q:i for i,q in enumerate(ids['query_ids'])}
rels=set()
with open(DATA/'qrels/test.tsv',newline='',encoding='utf8') as f:
 for row in csv.DictReader(f,delimiter='\t'):
  q=str(row['query-id'])
  if q in set(qids) and int(row['score'])>0: rels.add(q)
qids=[q for q in qids if q in rels]
Q=np.load(HERE/'runs/query_vectors.npy',allow_pickle=False)[np.asarray([qindex[q] for q in qids])]
lens=np.load(HERE/'runs/query_lengths.npy',allow_pickle=False)[np.asarray([qindex[q] for q in qids])]
roles=np.where(lens<=freeze['role_threshold_wordpieces'],'short','long')
searcher=scann.scann_ops_pybind.load_searcher(str(HERE/'payload/scann_index'))
angles=freeze['angles_rad']; arms=('native','mirror','linear_control')
# Warm each arm before timing and preserve independent observations per query/arm.
for arm in arms:
 i=0
 q=Q[i:i+1] if arm=='native' else (givens(Q[i:i+1],angles[roles[i]]) if arm=='mirror' else linear(Q[i:i+1],angles[roles[i]]))
 searcher.search(q[0],final_num_neighbors=10,pre_reorder_num_neighbors=100,leaves_to_search=16)
values={a:[] for a in arms}; rng=random.Random(1040)
for rep in range(5):
 for i in range(len(qids)):
  order=list(arms); rng.shuffle(order)
  for arm in order:
   start=time.perf_counter_ns()
   if arm=='native': q=Q[i]
   elif arm=='mirror': q=givens(Q[i:i+1],angles[roles[i]])[0]
   else: q=linear(Q[i:i+1],angles[roles[i]])[0]
   searcher.search(q,final_num_neighbors=10,pre_reorder_num_neighbors=100,leaves_to_search=16)
   values[arm].append((time.perf_counter_ns()-start)/1e6)
out={'repeats_per_query_arm':5,'queries':len(qids),'randomized_arm_order':True,'seed':1040,'latency_scope':'CPU query Givens/linear transform plus ScaNN single-query search; excludes tokenizer/encoder, includes dynamic view computation','cpu_affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,'rows':{a:{'p50_ms':float(np.percentile(v,50)),'p95_ms':float(np.percentile(v,95)),'mean_ms':float(np.mean(v)),'n':len(v)} for a,v in values.items()}}
(HERE/'runs/LATENCY_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
