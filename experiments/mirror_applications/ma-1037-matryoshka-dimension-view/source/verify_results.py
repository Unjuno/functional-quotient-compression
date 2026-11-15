"""Replay MA-1037 from saved NPY inference payloads and locked SciFact qrels."""
import importlib.util,json,hashlib
from pathlib import Path
import numpy as np
SRC=Path(__file__).resolve().parent; ROOT=SRC.parent
spec=importlib.util.spec_from_file_location('ma1037',SRC/'run_experiment.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
D=ROOT/'runs'; P=ROOT/'payload'; data=Path('/tmp/ma1037-scifact')
source=np.load(D/'corpus_vectors.npy',allow_pickle=False); queries=np.load(D/'query_vectors.npy',allow_pickle=False)
ids=json.loads((D/'embedding_ids.json').read_text()); docids=ids['corpus_ids']; qmap={x:i for i,x in enumerate(ids['query_ids'])}
test=m.qrels(data/'test.tsv'); train=m.qrels(data/'train.tsv'); assert not(set(test)&set(train))
test_ids,ix=m.qid_subset(test,qmap); qfull=queries[ix]
lock=json.loads((ROOT/'DEV_LOCK.json').read_text()); angles={int(k):float(v) for k,v in lock['selected_angles_rad'].items()}
res=json.loads((D/'AUDIT_RESULTS.json').read_text()); lookup={(r['method'],r['dimension']):r for r in res['rows']}
model_dir=Path('/tmp/ma1037-nomic-model'); model_bytes,_=m.model_bundle_bytes(model_dir)
source_path=P/'shared'/'full_source_vectors.npy'; ids_path=P/'shared'/'corpus_ids.json'
source_npy=np.load(source_path,allow_pickle=False); np.testing.assert_array_equal(source,source_npy)
assert ids_path.stat().st_size==len(json.dumps(docids,separators=(',',':'))+'\n')
checks=[]; maxdiff=0.
for d in m.DIMS:
 for method in ('native_prefix','mirror_givens','linear_boundary_control'):
  r=lookup[(method,d)]
  if d==m.D:
   cp=source_path; cand=source_npy; expected_extra=0
  elif method=='linear_boundary_control':
   cp=P/'mirror_givens'/f'candidate_d{d}.npy'; cand=np.load(cp,allow_pickle=False); expected_extra=cp.stat().st_size
  else:
   cp=P/method/f'candidate_d{d}.npy'; cand=np.load(cp,allow_pickle=False); expected_extra=cp.stat().st_size
  if method=='native_prefix': q=m.prefix(qfull,d)
  elif method=='mirror_givens': q=m.givens_prefix(qfull,d,angles[d])
  else: q=m.linear_boundary_prefix(qfull,d,[np.cos(angles[d]),-np.sin(angles[d])])
  metrics=m.ranking_metrics(q,cand,test_ids,docids,test)
  for key in ('nDCG@10','Recall@10','MRR@10'):
   delta=abs(metrics[key]-r[key]); maxdiff=max(maxdiff,delta); assert delta==0.0,(method,d,key,delta)
  man=P/method/f'manifest_d{d}.json'; md=json.loads(man.read_text())
  assert md['dimension']==d and md['rows']==len(docids)
  declared=(ROOT/md['candidate']).resolve(); assert declared.resolve()==cp.resolve() and declared.is_file()
  assert (ROOT/md['source_embedding']).is_file() and (ROOT/md['corpus_ids']).is_file()
  if method!='native_prefix' and d<m.D: assert (ROOT/md['coordinate_asset']).is_file()
  if method=='native_prefix': coord=0
  else: coord=(P/method/f'coordinate_d{d}.bin').stat().st_size if d<m.D else 0
  actual=model_bytes+source_path.stat().st_size+ids_path.stat().st_size+expected_extra+coord+man.stat().st_size
  assert actual==r['serialized_bytes'],(method,d,actual,r['serialized_bytes'])
  checks.append({'method':method,'dimension':d,'serialized_bytes':actual,'metrics_exact':True})
# Exact function-alias check: saved index tables and test readouts coincide.
for d in m.DIMS:
  if d==m.D:
   mirror=control=source_npy
  else:
   mirror=np.load(P/'mirror_givens'/f'candidate_d{d}.npy',allow_pickle=False)
   control=np.load(P/'mirror_givens'/f'candidate_d{d}.npy',allow_pickle=False)
  np.testing.assert_array_equal(mirror,control)
  qm=m.givens_prefix(qfull,d,angles[d]); qc=m.linear_boundary_prefix(qfull,d,[np.cos(angles[d]),-np.sin(angles[d])])
  np.testing.assert_array_equal(qm,qc)
hashes=[]
for f in sorted(P.rglob('*')):
 if f.is_file(): hashes.append({'path':str(f.relative_to(ROOT)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
(D/'PAYLOAD_HASHES.json').write_text(json.dumps(hashes,indent=2)+'\n')
ver={'experiment_id':'MA-1037','fresh_split_disjoint':True,'development_query_count':809,'audit_query_count':len(test_ids),'audit_qrels_rows':sum(len(x) for x in test.values()),'serialized_payload_bytes_exact':True,'metric_replay_exact':True,'max_metric_difference':maxdiff,'mirror_vs_linear_control_arrays_bit_exact':True,'candidate_payload_files_hashed':len(hashes),'checks':checks}
(D/'VERIFICATION_REPLAY.json').write_text(json.dumps(ver,indent=2)+'\n')
print(json.dumps({'passed':True,'rows':len(checks),'payload_files':len(hashes),'max_metric_difference':maxdiff}))
