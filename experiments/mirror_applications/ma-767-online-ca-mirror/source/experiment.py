"""MA-767: online adaptation of synthetic binary CA local rules."""
from __future__ import annotations
import csv,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from safetensors.torch import save_file
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(1)
CHECKPOINTS=[0,8,32,64,128,256]

def make_world(seed):
 rng=np.random.default_rng(seed);torch.manual_seed(seed)
 base=rng.normal(0,1.5,(8,)).astype('float32');q,_=np.linalg.qr(rng.normal(size=(8,2)));true_basis=q[:,:2].astype('float32')
 dev=[base+true_basis@rng.normal(0,.5,2) for _ in range(4)];mat=np.stack([x-base for x in dev]);_,_,vt=np.linalg.svd(mat,full_matrices=False);basis=vt[:2].T.astype('float32')
 tasks=[]
 for i in range(32):tasks.append(('easy',base+true_basis@rng.normal(0,.55,2).astype('float32')))
 for i in range(32):tasks.append(('hard',base+rng.normal(0,1.5,8).astype('float32')))
 rng.shuffle(tasks)
 return base,basis,dev,tasks,rng

def observations(rng,logits,n):
 z=rng.integers(0,8,size=n,dtype=np.int64);p=1/(1+np.exp(-logits[z]));y=rng.binomial(1,p).astype('float32');return z,y

def acc(logits,z,y):return float(np.mean((logits[z]>=0)==(y>=.5)))
def adapt_one(kind,base,B,trainz,trainy,queryz,queryy,lr=.08):
 if kind=='mirror':state=torch.zeros(2,requires_grad=True);basis=torch.tensor(B);theta=torch.tensor(base)
 else:state=torch.tensor(base.copy(),requires_grad=True);basis=None;theta=None
 opt=torch.optim.SGD([state],lr=lr);metrics={0:acc(base,queryz,queryy)};start=time.perf_counter();last=0
 for step,(z,y) in enumerate(zip(trainz,trainy),1):
  ix=torch.tensor([int(z)]);target=torch.tensor([float(y)])
  pred=(theta+basis@state)[ix] if kind=='mirror' else state[ix]
  loss=torch.nn.functional.binary_cross_entropy_with_logits(pred,target);opt.zero_grad();loss.backward();opt.step()
  if step in CHECKPOINTS:
   rule=(base+B@state.detach().numpy()) if kind=='mirror' else state.detach().numpy()
   metrics[step]=acc(rule,queryz,queryy)
  last=step
 return state.detach().numpy(),metrics,time.perf_counter()-start

def pack(method,base,B,states,path):
 data={}
 if method=='mirror':data['base']=torch.tensor(base).contiguous();data['basis']=torch.tensor(B).contiguous()
 if method=='mirror':data['code_bank']=torch.stack([torch.as_tensor(v,dtype=torch.float32).reshape(-1) for v in states]).contiguous()
 elif method=='full_rule_bank':data['rule_bank']=torch.stack([torch.as_tensor(v,dtype=torch.float32).reshape(-1) for v in states]).contiguous()
 else:data['active_rule']=torch.as_tensor(states[0],dtype=torch.float32).reshape(-1).contiguous()
 save_file(data,str(path));raw=path.read_bytes();return len(raw),hashlib.sha256(raw).hexdigest()

def main():
 cfg=json.loads((ROOT/'PROTOCOL.json').read_text());rows=[]
 for seed in cfg['fresh']['worlds_or_seeds']:
  base,B,dev,tasks,rng=make_world(seed);m_codes=[];f_rules=[];per=[];overwrite_rules=[];testsets=[]
  for ti,(stratum,teacher) in enumerate(tasks):
   tr=observations(rng,teacher,256);te=observations(rng,teacher,512);testsets.append(te)
   m,mm,mt=adapt_one('mirror',base,B,*tr,*te);f,fm,ft=adapt_one('full',base,B,*tr,*te);m_codes.append(m);f_rules.append(f);overwrite_rules.append(f)
   for step in CHECKPOINTS:
    per.append({'world':seed,'task':ti,'stratum':stratum,'method':'mirror','online_examples':step,'accuracy':mm[step],'rule_logit_mse':float(np.mean((base+B@m-teacher)**2)) if step==256 else None,'update_seconds':mt*step/256,'updates':step,'active_macs_per_prediction':18,'retention_ratio':None})
    per.append({'world':seed,'task':ti,'stratum':stratum,'method':'full_rule_bank','online_examples':step,'accuracy':fm[step],'rule_logit_mse':float(np.mean((f-teacher)**2)) if step==256 else None,'update_seconds':ft*step/256,'updates':step,'active_macs_per_prediction':1,'retention_ratio':None})
  # Frozen shared control and sequential overwrite retention; stored banks retain each task code/rule unchanged.
  for ti,(stratum,teacher) in enumerate(tasks):
   z,y=observations(rng,teacher,512);per.append({'world':seed,'task':ti,'stratum':stratum,'method':'frozen_shared','online_examples':0,'accuracy':acc(base,z,y),'rule_logit_mse':float(np.mean((base-teacher)**2)),'update_seconds':0.0,'updates':0,'active_macs_per_prediction':1})
   per.append({'world':seed,'task':ti,'stratum':stratum,'method':'full_overwrite_final','online_examples':256,'accuracy':acc(f_rules[-1],z,y),'rule_logit_mse':float(np.mean((f_rules[-1]-teacher)**2)),'update_seconds':0.0,'updates':256,'active_macs_per_prediction':1})
  # Explicit post-adaptation bank retention: evaluate each stored task view after all tasks have been processed.
  for ti,(stratum,teacher) in enumerate(tasks):
   z,y=testsets[ti];ma_rule=base+B@m_codes[ti];fu_rule=f_rules[ti]
   ma_acc=acc(ma_rule,z,y);fu_acc=acc(fu_rule,z,y)
   per.append({'world':seed,'task':ti,'stratum':stratum,'method':'mirror_bank_after_all','online_examples':256,'accuracy':ma_acc,'rule_logit_mse':float(np.mean((ma_rule-teacher)**2)),'update_seconds':0.0,'updates':256,'active_macs_per_prediction':18,'private_active_fraction':0.0,'retention_ratio':1.0})
   per.append({'world':seed,'task':ti,'stratum':stratum,'method':'full_rule_bank_after_all','online_examples':256,'accuracy':fu_acc,'rule_logit_mse':float(np.mean((fu_rule-teacher)**2)),'update_seconds':0.0,'updates':256,'active_macs_per_prediction':1,'private_active_fraction':1.0,'retention_ratio':1.0})
  # Actual serialized inference payloads: shared state once and all task codes/snapshots.
  for method,states in [('mirror',m_codes),('full_rule_bank',f_rules),('full_overwrite',[f_rules[-1]])]:
   path=ROOT/'source'/f'.{method}-{seed}.safetensors';n,h=pack(method,base,B,states,path);path.unlink()
   for r in per:
    if r['method'] in (('full_overwrite_final',) if method=='full_overwrite' else ((method,method+'_after_all'))):r['serialized_bytes']=n;r['payload_sha256']=h
  rows.extend(per)
  for st in ['easy','hard']:
   print(seed,st,flush=True)
   for meth in ['mirror','full_rule_bank','frozen_shared','full_overwrite_final']:
    q=[r for r in per if r['stratum']==st and r['method']==meth and (r['online_examples']==64 if meth in ('mirror','full_rule_bank') else True)]
    if q:print(meth,'acc',float(np.mean([x['accuracy'] for x in q])),'bytes',q[0].get('serialized_bytes',''),flush=True)
 (ROOT/'source'/'audit_results.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  fields=sorted(set().union(*(r.keys() for r in rows)));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
