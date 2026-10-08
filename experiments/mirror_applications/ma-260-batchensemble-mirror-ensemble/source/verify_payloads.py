"""Replay member metrics from all serialized development/fresh payloads."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'))
import run_experiment as exp

def verify():
 rows=[]
 for split,seeds in (('development',(26001,26002)),('fresh',(26011,26012,26013))):
  for regime in ('aligned','unrelated'):
   for seed in seeds:
    d=json.loads((ROOT/f'results/{split}'/f'{regime}_{seed}_result.json').read_text());_w,_a,_tr,test=exp.make_world(seed,regime);xt,yt,tp=test
    for method,m in d['methods'].items():
     path=ROOT/f'results/{split}'/f'{regime}_{seed}_{method}.zip';blob=path.read_bytes();assert len(blob)==m['actual_payload_bytes'] and hashlib.sha256(blob).hexdigest()==m['payload_sha256']
     _meta,_arr,w,b=exp.decode(path);q=exp.metrics(w,b,xt,yt,tp);delta=max(abs(q[k]-m['metrics'][k]) for k in ('mean_member_nll','mean_member_accuracy','mean_member_ece','mixture_nll','mixture_accuracy','mean_pairwise_JS_divergence'));assert delta<1e-7,(seed,regime,method,delta)
     rows.append({'split':split,'seed':seed,'regime':regime,'method':method,'bytes':len(blob),'sha256':m['payload_sha256'],'metric_max_difference':delta})
 return rows
if __name__=='__main__':print(json.dumps({'payloads_replayed':len(verify()),'rows':verify()},indent=2))
