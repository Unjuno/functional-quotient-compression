import hashlib,json
from pathlib import Path
import torch
from model import data_splits,load_flat,DigitsMLP,metrics,posterior_logits,rank1_logits,disagreement
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'source';raw=json.loads((SRC/'development_raw.json').read_text());xtr,ytr,xdev,ydev,xaudit,yaudit=data_splits()
assert not raw['fresh_accessed'] and not raw['audit_labels_used']
def shifted(x,seed):
 g=torch.Generator().manual_seed(seed);return (x+0.25*torch.randn(x.shape,generator=g)).clamp(0,1)
payload_map={};objs={}
if not (SRC/'payloads').exists() or not any((SRC/'payloads').glob('*.pt')):
 raise SystemExit('payload captures are external/ignored; rerun source/run_experiment.py with the frozen environment, then verify')
for p in raw['payloads']:
 path=ROOT/p['path'];b=path.read_bytes();assert len(b)==p['bytes'];assert hashlib.sha256(b).hexdigest()==p['sha256'];payload_map[p['key']]=p;objs[p['key']]=torch.load(path,map_location='cpu',weights_only=True)
maxdiff=0.;checked=0
for row in raw['rows']:
 x=xdev if row['condition']=='clean' else shifted(xdev,row['seed']+880);key=row['payload_key'];obj=objs[key];y=ydev
 if row['method']=='swa_mean':
  logits=load_flat(DigitsMLP(),obj['mean'])(x);members=None
 elif row['method']=='independent_ensemble':
  members=torch.stack([load_flat(DigitsMLP(),q)(x).softmax(-1) for q in obj['members']]);logits=members.mean(0).clamp_min(1e-12).log()
 elif row['method'] in ('swag_gaussian','mirror_rademacher','gaussian_lowrank_only'):
  logp,members=posterior_logits(obj['mean'],obj['basis'],obj['eigen_scales'],obj['diagonal_std'],x,obj['sample_seed'],obj['posterior_members'],row['method']=='mirror_rademacher');logits=logp
 elif row['method']=='rank1_factor':
  logits,members=rank1_logits(obj['mean'],x,obj['sample_seed'],obj['global_factor_sigma'],obj['posterior_members'])
 else:raise ValueError(row['method'])
 got=metrics(logits,y)
 if members is not None:got['pairwise_disagreement']=disagreement(members)
 for metric,value in row['metrics'].items():
  delta=abs(got[metric]-value);maxdiff=max(maxdiff,delta)
  if delta>1e-6:raise SystemExit(f"metric replay failed {metric}: {delta}, {key}")
 checked+=1
rep={'checked':True,'rows':checked,'serialized_payloads':len(payload_map),'payload_hashes_and_actual_lengths_checked':True,'metrics':['NLL','accuracy','ECE10','Brier','pairwise disagreement'],'max_abs_metric_delta':maxdiff,'fresh_accessed':False,'audit_labels_used':False}
(SRC/'metric_replay.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep,indent=2))
