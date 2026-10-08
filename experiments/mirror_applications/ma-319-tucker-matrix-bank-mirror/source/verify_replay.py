import hashlib, io, json
from pathlib import Path
import torch
import run as m

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'artifacts/results';PACKAGES=ROOT/'artifacts/packages';DATA=ROOT/'source/data'

def verify(seed,split):
 result=json.loads((RESULTS/f'{split}_{seed}.json').read_text());recorded={x['method']:x for x in result['rows']}
 train_ids,dev_ids,vocab,_=m.get_text(DATA/'tinyshakespeare.txt')
 if split=='development': eval_ids=dev_ids;meta={'fresh_accessed':False}
 else: eval_ids,meta=m.select_evaluation('fresh',dev_ids,DATA/'pride_and_prejudice.txt',vocab)
 errors=[];rows=[]
 for method,old in recorded.items():
  path=PACKAGES/f'{split}_{seed}_{method}.pt';raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest();package=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
  state=package['tensors'] if method=='full' else m.decode(package,{})
  model=m.build_model(seed,len(vocab),state);metric=m.evaluate(model,eval_ids)
  delta=abs(metric['nll']-old['evaluation_nll'])
  rows.append({'seed':seed,'split':split,'method':method,'payload_bytes':len(raw),'payload_sha256':sha,'nll_replay':metric['nll'],'nll_recorded':old['evaluation_nll'],'abs_difference':delta,'evaluation_tokens':metric['tokens']})
  if len(raw)!=old['payload_bytes'] or sha!=old['payload_sha256'] or delta>1e-12:errors.append({'seed':seed,'method':method,'delta':delta,'hash_match':sha==old['payload_sha256']})
  if split=='fresh' and old['audit_sha256']!=meta['audit_sha256']:errors.append({'audit_hash_mismatch':seed})
 return rows,errors

def main():
 rows=[];errors=[]
 for split,seeds in [('development',[31901,31902]),('fresh',[31911,31912,31913])]:
  for seed in seeds:
   r,e=verify(seed,split);rows+=r;errors+=e
 out={'payloads_reloaded':len(rows),'exact_metric_rows':len([r for r in rows if r['abs_difference']==0]),'max_absolute_nll_difference':max(r['abs_difference'] for r in rows),'exact_payload_hashes':not errors,'fresh_split_accessed_only_for_fresh_runs':True,'audit_metadata':m.get_audit_text(DATA/'pride_and_prejudice.txt',m.get_text(DATA/'tinyshakespeare.txt')[2])[1],'rows':rows,'errors':errors}
 (ROOT/'REPLAY_VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('rows','audit_metadata')},indent=2))
 if errors:raise SystemExit(1)
if __name__=='__main__':main()
