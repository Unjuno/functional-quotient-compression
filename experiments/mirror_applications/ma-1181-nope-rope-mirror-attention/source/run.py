"""MA-1181 NoPE/RoPE x shared/FiLM/Mirror placement with frozen protocol."""
from __future__ import annotations
import argparse, csv, hashlib, io, json, os, time, platform
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BASE=Path(__file__).resolve().parent
PROTOCOL=json.loads((BASE/'PROTOCOL.json').read_text())
SEEDS=PROTOCOL['design']['seeds']
POSITIONS=PROTOCOL['models']['position']
ARMS=PROTOCOL['models']['arms']
TASKS=list(PROTOCOL['tasks'])
VOCAB=13; D=16; N_PREFIX=9; T=10

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,d):Path(path).write_text(json.dumps(d,indent=2,ensure_ascii=False,sort_keys=True)+'\n')

def reflect(x,v):
    v=v/(v.square().sum().sqrt().clamp_min(1e-10))
    return x-2*((x*v).sum(-1,keepdim=True))*v

def rope(x):
    # x has shape batch x length x d; exact canonical rotation of Q & K after Q-Mirror
    dev=x.device
    pos=torch.arange(x.shape[-2],device=dev,dtype=x.dtype)
    inv=10000**(-torch.arange(0,D,2,device=dev,dtype=x.dtype)/D)
    phase=pos[:,None]*inv[None,:]
    s,c=torch.sin(phase),torch.cos(phase)
    a,b=x[...,::2],x[...,1::2]
    return torch.stack((c*a-s*b,s*a+c*b),dim=-1).flatten(-2)

class RecurrentBlock(nn.Module):
 def __init__(self):
  super().__init__()
  self.norm1=nn.LayerNorm(D)
  self.qkv=nn.Linear(D,3*D,bias=False)
  self.proj=nn.Linear(D,D)
  self.norm2=nn.LayerNorm(D)
  self.ff=nn.Sequential(nn.Linear(D,32),nn.SiLU(),nn.Linear(32,D))
 def forward(self,h,pos,arm,code):
  y=self.norm1(h)
  q,k,v=self.qkv(y).chunk(3,dim=-1)
  if arm=='mirror_q':q=reflect(q,code)
  if arm=='film_q':q=q*(1+0.4*torch.tanh(code))
  if pos=='rope':q,k=rope(q),rope(k)
  score=q@k.transpose(-2,-1)/(D**0.5)
  mask=torch.ones(T,T,device=h.device,dtype=torch.bool).triu(1)
  score=score.masked_fill(mask,float('-inf'))
  att=torch.softmax(score,dim=-1)
  h=h+self.proj(att@v)
  h=h+self.ff(self.norm2(h))
  return h

class Model(nn.Module):
 def __init__(self,arm,pos,seed):
  super().__init__()
  assert arm in ARMS and pos in POSITIONS
  self.arm=arm;self.pos=pos
  # Match same base-model RNG for all arms/positions/task at same seed.
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(seed)
   self.emb=nn.Embedding(VOCAB,D)
   self.block=RecurrentBlock()
   self.norm=nn.LayerNorm(D)
   self.head=nn.Linear(D,2)
  if arm!='shared':
   with torch.random.fork_rng(devices=[]):
    torch.manual_seed(seed+100000)
    self.codes=nn.Parameter(torch.randn(2,D)*0.1 if arm.startswith('mirror') else torch.zeros(2,D))
 def forward(self,x):
  h=self.emb(x)
  for step in range(2):
   if self.arm=='mirror_residual':
    h=reflect(h,self.codes[step]);h=self.block(h,self.pos,self.arm,None);h=reflect(h,self.codes[step])
   else:
    h=self.block(h,self.pos,self.arm,self.codes[step] if self.arm!='shared' else None)
  return self.head(self.norm(h[:,-1]))

def data(task,seed,n,split):
  # deterministic split and full distribution regeneration; explicit balancing
  rng=np.random.default_rng(781262+seed*10003+split*911+TASKS.index(task)*100001)
  labels=rng.integers(0,2,size=n,dtype=np.int64)
  x=rng.integers(0,12,size=(n,N_PREFIX),dtype=np.int64)
  for i,y in enumerate(labels):
   x[i]=rng.choice([t for t in range(12) if t!=3],size=N_PREFIX)
   if task=='set_membership':
    if y:x[i,rng.integers(N_PREFIX)]=3
   else:
    x[i,N_PREFIX-1 if y else rng.integers(N_PREFIX-1)]=3
  inp=np.concatenate((x,np.full((n,1),12,dtype=np.int64)),axis=1)
  return torch.from_numpy(inp),torch.from_numpy(labels)

def do_train(arm,pos,task,seed,out):
  cfg=PROTOCOL['design'];m=Model(arm,pos,seed)
  x,y=data(task,seed,cfg['n_train'],1)
  dx,dy=data(task,seed,cfg['n_dev'],2)
  g=torch.Generator().manual_seed(987122+seed+TASKS.index(task)*100)
  batches=torch.randint(len(x),(cfg['steps'],cfg['batch']),generator=g)
  opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['adamw_wd'])
  best=(float('inf'),None,None); tic=time.perf_counter();hist={}
  for step in range(1,cfg['steps']+1):
   ind=batches[step-1]
   l=F.cross_entropy(m(x[ind]),y[ind]);opt.zero_grad(set_to_none=True);l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),cfg['clip_norm']);opt.step()
   if step in cfg['checkpoint_selection']:
    with torch.no_grad():dv=F.cross_entropy(m(dx),dy).item()
    hist[str(step)]=dv
    if dv<best[0]:best=(dv,{k:v.detach().clone() for k,v in m.state_dict().items()},step)
  wall=time.perf_counter()-tic;m.load_state_dict(best[1]);m.eval()
  key=f'{task}_{pos}_{arm}_seed{seed}'
  ck=out/'checkpoints'/f'{key}.pt'
  torch.save({'state':best[1],'arm':arm,'pos':pos,'seed':seed,'task':task,'step':best[2]},ck)
  return {'key':key,'arm':arm,'pos':pos,'task':task,'seed':seed,'dev_ce':best[0],'dev_curve':hist,'step':best[2],'train_wall_s':wall,'params':sum(p.numel() for p in m.parameters()),'checkpoint':ck.name,'sha256':sha(ck)}

def latency(model):
 with torch.no_grad():
  x=torch.randint(0,12,(64,T));x[:,-1]=12
  for _ in range(8):model(x)
  tic=time.perf_counter()
  for _ in range(30):model(x)
  return (time.perf_counter()-tic)*1e6/30

def main(out):
 torch.set_num_threads(1)
 out.mkdir(parents=True,exist_ok=False)
 (out/'checkpoints').mkdir()
 (out/'PROTOCOL.json').write_bytes((BASE/'PROTOCOL.json').read_bytes())
 rows=[]; start=time.perf_counter()
 for task in TASKS:
  for seed in SEEDS:
   for pos in POSITIONS:
    for arm in ARMS:
     row=do_train(arm,pos,task,seed,out)
     rows.append(row)
     print('TRAIN',len(rows),'/',48,row['key'],'dev',round(row['dev_ce'],4),flush=True)
 assert len(rows)==PROTOCOL['design']['expected_students']
 write_json(out/'SEALED_MANIFEST.json',{'id':'MA-1181','protocol_sha':sha(BASE/'PROTOCOL.json'),'fresh_generated':False,'records':rows,'start_wall_s':time.perf_counter()-start,'torch':torch.__version__,'python':platform.python_version(),'cpu':platform.processor(),'threads':torch.get_num_threads()})
 # Only now create fresh input splits
 scored=[]
 for row in rows:
  assert sha(out/'checkpoints'/row['checkpoint'])==row['sha256']
  checkpoint=torch.load(out/'checkpoints'/row['checkpoint'],weights_only=True)
  m=Model(row['arm'],row['pos'],row['seed']);m.load_state_dict(checkpoint['state']);m.eval()
  fx,fy=data(row['task'],row['seed'],PROTOCOL['design']['n_fresh'],3)
  with torch.no_grad():
   logits=m(fx)
   ce=F.cross_entropy(logits,fy).item()
   acc=(logits.argmax(-1)==fy).float().mean().item()
  p=io.BytesIO();torch.save({'arch':'MA1181-tiny-transformer','pos':row['pos'],'arm':row['arm'],'state':m.state_dict()},p)
  scored.append({**row,'fresh_ce':ce,'fresh_acc':acc,'packet_bytes':len(p.getvalue()),'latency_b64_us':latency(m)})
 write_json(out/'FULL_RESULTS.json',scored)
 with open(out/'RESULTS_CORE.csv','w',newline='') as f:
  field=[k for k in scored[0] if k!='dev_curve'];wr=csv.DictWriter(f,fieldnames=field);wr.writeheader();wr.writerows(({k:v for k,v in row.items() if k!='dev_curve'} for row in scored))
 from collections import defaultdict
 grouped=defaultdict(dict)
 for r in scored:grouped[(r['task'],r['seed'],r['pos'])][r['arm']]=r
 assessment=[]
 for task in TASKS:
  for seed in SEEDS:
   a=grouped[(task,seed,'nope')];b=grouped[(task,seed,'rope')]
   gap_nope=a['film_q']['fresh_ce']-a['mirror_q']['fresh_ce']
   gap_rope=b['film_q']['fresh_ce']-b['mirror_q']['fresh_ce']
   assessment.append({'task':task,'seed':seed,'nope_gap':gap_nope,'rope_gap':gap_rope,'interaction':gap_nope-gap_rope,'primary_screening_pass':gap_nope-gap_rope>=.02,'mirror_nope_useful':gap_nope>=.02})
 from statistics import median
 write_json(out/'ANALYSIS.json',{'assessments':assessment,'primary_passes':sum(a['primary_screening_pass'] for a in assessment if a['task']=='set_membership'),'mirror_nope_passes':sum(a['mirror_nope_useful'] for a in assessment if a['task']=='set_membership'),'condition_medians':{str((t,p,a)):median(r['fresh_ce'] for r in scored if r['task']==t and r['pos']==p and r['arm']==a) for t in TASKS for p in POSITIONS for a in ARMS},'result_hash':sha(out/'RESULTS_CORE.csv')})
 # Replay fresh exactness on all 48 models
 maxerr=0
 for r in scored:
  ck=out/'checkpoints'/r['checkpoint'];assert sha(ck)==r['sha256']
  m=Model(r['arm'],r['pos'],r['seed']);m.load_state_dict(torch.load(ck,weights_only=True)['state']);m.eval()
  fx,fy=data(r['task'],r['seed'],PROTOCOL['design']['n_fresh'],3)
  with torch.no_grad(): rec=F.cross_entropy(m(fx),fy).item()
  maxerr=max(maxerr,abs(rec-r['fresh_ce']))
 write_json(out/'VERIFICATION.json',{'checkpoint_hash_ok':len(rows),'fresh_replays':len(scored),'max_fresh_ce_diff':maxerr,'manifest_fresh_generated':json.loads((out/'SEALED_MANIFEST.json').read_text())['fresh_generated'],'total_wall_seconds':time.perf_counter()-start})
 print('DONE',len(rows),'total_s',time.perf_counter()-start,'max_replay_error',maxerr,flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();main(args.out)
