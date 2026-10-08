import csv,hashlib,json
from pathlib import Path
import torch
from model import unitary,unitary_metrics,make_tasks
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'source';ART=SRC/'artifacts';raw=json.loads((SRC/'development_raw.json').read_text())
libs={}
for x in raw['libraries']:
 p=ROOT/x['path'];b=p.read_bytes();assert len(b)==x['bytes'];assert hashlib.sha256(b).hexdigest()==x['sha256']
 libs[(x['seed'],x['rho'],x['method'],x['rank'])]=torch.load(p,map_location='cpu',weights_only=True)
maxdiff=0.0
for row in raw['rows']:
 key=(row['seed'],row['rho'],row['method'],row['rank']);obj=libs[key];task=row['task_id']
 cp=ART/f"cond_seed{row['seed']}_rho{row['rho']}_{row['method']}_r{row['rank']}_task{task}.pt";cb=cp.read_bytes()
 assert len(cb)==row['condition_state_bytes'];assert hashlib.sha256(cb).hexdigest()==row['condition_state_sha256']
 condition=torch.load(cp,map_location='cpu',weights_only=True);assert condition['task_id']==task
 if row['method']=='independent':angle=obj['angles'][task]
 elif row['method']=='hard_shared':angle=obj['mean']
 elif row['method']=='matrix_lowrank':angle=obj['mean']+torch.einsum('r,rlqa->lqa',obj['codes'][task],obj['basis'])
 elif row['method']=='tt_hyper':
  from model import tt_reconstruct
  angle=obj['mean']+tt_reconstruct(obj['codes'][task],obj['cores'])
 else:
  angle=obj['mean']+torch.einsum('r,lr,qr,ar->lqa',obj['codes'][task],obj['layer_factors'],obj['qubit_factors'],obj['axis_factors'])
  if row['method']=='mirror_cp_private':
   res=torch.zeros(72);res[obj['private_indices'][task]]=obj['private_values'][task];angle=angle+res.reshape(6,4,3)
 target,_=make_tasks(row['seed'],row['rho']);u0=unitary(target[task].numpy());u1=unitary(angle.double().numpy());f,_=unitary_metrics(u0,u1);d=abs(f-row['process_fidelity']);maxdiff=max(maxdiff,d)
 if d>1e-10:raise SystemExit(f"process fidelity replay mismatch: {key}, task={task}, delta={d}")
rep={'checked':True,'rows':len(raw['rows']),'library_payloads':len(raw['libraries']),'condition_payloads':len(raw['rows']),'serialized_sizes_and_sha256_checked':True,'process_fidelity_max_difference':maxdiff,'fresh_accessed':False,'qpu_or_shots_used':False}
(SRC/'metric_replay.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep,indent=2))
