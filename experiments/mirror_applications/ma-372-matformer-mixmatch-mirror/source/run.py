from __future__ import annotations
import argparse,csv,hashlib,io,json,random,time
from pathlib import Path
import numpy as np,sklearn,torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from model import WIDTHS,CONFIGS,MIXED,METHODS,Nested3,macs,transform
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';UPDATES=800;CODE_UPDATES=100;BATCH=64
FIELDS=['condition','world_or_seed','learning_rate','method','configuration','serialized_bytes','train_examples_seen','optimizer_updates','active_MAC_proxy','adapt_MAC_proxy','train_wall_time_s','inference_examples_per_second','accuracy','nll','payload_sha256','roundtrip_max_abs_diff','status_note']
HASH='faf2d98f61250c1cdc0b114323133d3c58da04f5c5d28bb78e4bde3c936a75b1'
def seedall(s):random.seed(s);np.random.seed(s);torch.manual_seed(s)
def world(s):
 d=load_digits();h=hashlib.sha256(d.data.astype('float32').tobytes()+d.target.astype('int64').tobytes()).hexdigest()
 if h!=HASH or sklearn.__version__!='1.8.0':raise RuntimeError('dataset provenance mismatch')
 ids=np.arange(len(d.target));tr,rest=train_test_split(ids,train_size=.6,stratify=d.target,random_state=s);va,te=train_test_split(rest,test_size=.5,stratify=d.target[rest],random_state=s+1)
 return {'x':torch.tensor(d.data.astype('float32')/16),'y':torch.tensor(d.target.astype('int64')),'tr':torch.tensor(tr),'va':torch.tensor(va),'te':torch.tensor(te)}
def batch(w,g):
 ids=w['tr'][torch.randint(len(w['tr']),(BATCH,),generator=g)];return w['x'][ids],w['y'][ids]
def fit_base(w,seed,lr):
 seedall(seed);m=Nested3();opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);g=torch.Generator().manual_seed(seed+111);t=time.perf_counter();m.train()
 for _ in range(UPDATES):
  x,y=batch(w,g);teacher=m(x,(64,64,64));loss=F.cross_entropy(teacher,y)
  cfg=tuple(WIDTHS[int(torch.randint(0,3,(),generator=g))] for _ in range(3));z=m(x,cfg);loss+=F.cross_entropy(z,y)+.25*F.kl_div(F.log_softmax(z/2,-1),F.softmax(teacher.detach()/2,-1),reduction='batchmean')*4
  opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 return m.eval(),time.perf_counter()-t,UPDATES*BATCH*2,UPDATES*BATCH*(macs((64,64,64))+macs((32,16,64)))
def fit_factor(m,w,seed,kind):
 bank={};duration=0.;
 for width in WIDTHS:
  codes=[torch.nn.Parameter(torch.zeros(4) if kind=='factor_mirror' else torch.ones(4)) for _ in range(3)];opt=torch.optim.Adam(codes,lr=.01);gen=torch.Generator().manual_seed(seed+width+300);t=time.perf_counter()
  for p in m.parameters():p.requires_grad_(False)
  for _ in range(CODE_UPDATES):
   x,y=batch(w,gen);z=m(x,(width,)*3,kind,codes);loss=F.cross_entropy(z,y);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
  for p in m.parameters():p.requires_grad_(True)
  duration+=time.perf_counter()-t;bank[width]=[c.detach().clone() for c in codes]
 return bank,duration,CODE_UPDATES*BATCH*3

def fit_direct(m,w,seed):
 bank={};dur=0.;gen=torch.Generator().manual_seed(seed+999);xall=w['x'][w['tr']];yall=w['y'][w['tr']]
 for ci,cfg in enumerate(MIXED):
  codes=[torch.nn.Parameter(torch.zeros(4)) for _ in range(3)];opt=torch.optim.Adam(codes,lr=.01);t=time.perf_counter()
  for p in m.parameters():p.requires_grad_(False)
  for _ in range(CODE_UPDATES):
   ids=torch.randint(len(yall),(BATCH,),generator=gen);x,y=xall[ids],yall[ids];z=m(x,cfg,'direct_mirror',codes);loss=F.cross_entropy(z,y);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
  for p in m.parameters():p.requires_grad_(True)
  dur+=time.perf_counter()-t;bank[cfg]=[c.detach().clone() for c in codes]
 return bank,dur,CODE_UPDATES*BATCH*len(MIXED)
def predict(m,bank,method,cfg,x):
 if method=='nested':return m(x,cfg)
 if method in ('factor_mirror','factor_film'):
  codes=[bank[width][layer] for layer,width in enumerate(cfg)];return m(x,cfg,method,codes)
 return m(x,cfg,'direct_mirror',bank[cfg])
def bundle(method,m,bank,meta):
 obj={'method':method,'meta':meta,'state':None if method=='direct_mirror' else {k:v.detach().cpu() for k,v in m.state_dict().items()},'direct_state':{k:v.detach().cpu() for k,v in m.state_dict().items()} if method=='direct_mirror' else None,'bank':{str(k):[v.detach().cpu() for v in vs] for k,vs in bank.items()}}
 bio=io.BytesIO();torch.save(obj,bio);return bio.getvalue()
def reload(data):
 obj=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False);m=Nested3();m.load_state_dict(obj['state'] if obj['state'] is not None else obj['direct_state']);m.eval();method=obj['method'];bank={}
 for k,vs in obj['bank'].items():
  key=tuple(map(int,k.strip('() ').split(','))) if ',' in k else int(k);bank[key]=vs
 return m,bank,method
def ev(data,w,part,cfg):
 m,bank,method=reload(data);ids=w[part];x,y=w['x'][ids],w['y'][ids]
 with torch.no_grad():
  z=predict(m,bank,method,cfg,x);nll=float(F.cross_entropy(z,y));acc=float((z.argmax(-1)==y).float().mean());t=time.perf_counter()
  for _ in range(20):predict(m,bank,method,cfg,x[:256])
  tps=20*min(len(x),256)/max(time.perf_counter()-t,1e-9)
 return acc,nll,tps
def runworld(seed,lr,phase):
 w=world(seed);s=seed+20000;m,base_sec,views,base_mac=fit_base(w,s,lr);fm,fmsec,fmviews=fit_factor(m,w,s,'factor_mirror');ff,ffsec,ffviews=fit_factor(m,w,s,'factor_film');direct,dsec,dviews=fit_direct(m,w,s);banks={'nested':{},'factor_mirror':fm,'factor_film':ff,'direct_mirror':direct};rows=[];meta={'world':seed,'lr':lr,'configs':CONFIGS,'fit_only_homogeneous':True}
 for method in METHODS:
  bank=banks[method];payload=bundle(method,m,bank,meta);digest=hashlib.sha256(payload).hexdigest();path=OUT/'payloads'/phase/str(seed)/str(lr)/(method+'.pt');path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
  if method=='nested':sec=base_sec;seen=views;updates=UPDATES;active=base_mac;adapt=0
  elif method=='factor_mirror':sec=base_sec+fmsec;seen=views+fmviews;updates=UPDATES+3*CODE_UPDATES;active=base_mac+fmviews*(macs((32,32,32))+72);adapt=fmviews*(macs((32,32,32))+72)
  elif method=='factor_film':sec=base_sec+ffsec;seen=views+ffviews;updates=UPDATES+3*CODE_UPDATES;active=base_mac+ffviews*(macs((32,32,32))+24);adapt=ffviews*(macs((32,32,32))+24)
  else:sec=base_sec+dsec;seen=views+dviews;updates=UPDATES+24*CODE_UPDATES;active=base_mac+dviews*(macs((32,32,32))+72);adapt=dviews*(macs((32,32,32))+72)
  # all 27 config descriptors charged in metadata; score all 24 mixed only
  diag=torch.linspace(0,1,4*64).reshape(4,64);m2,b2,method2=reload(payload);rd=0.
  for cfg in MIXED:rd=max(rd,float((predict(m,bank,method,cfg,diag)-predict(m2,b2,method2,cfg,diag)).detach().abs().max()))
  if rd>0:raise RuntimeError(f'roundtrip mismatch {rd}')
  for cfg in MIXED:
   acc,nll,tps=ev(payload,w,'va' if phase=='development' else 'te',cfg);rows.append({'condition':phase,'world_or_seed':seed,'learning_rate':lr,'method':method,'configuration':'-'.join(map(str,cfg)),'serialized_bytes':len(payload),'train_examples_seen':seen,'optimizer_updates':updates,'active_MAC_proxy':macs(cfg)+(0 if method=='nested' else 24 if method=='factor_film' else 72),'adapt_MAC_proxy':adapt,'train_wall_time_s':round(sec,6),'inference_examples_per_second':round(tps,3),'accuracy':f'{acc:.8f}','nll':f'{nll:.10f}','payload_sha256':digest,'roundtrip_max_abs_diff':rd,'status_note':f'{phase}/{seed}/{lr}/{method}; held-out mixed config'})
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=('development','fresh'),required=True);p.add_argument('--confirm-fresh',action='store_true');a=p.parse_args();torch.set_num_threads(1)
 if a.phase=='development':seeds=(37200,37201);lrs=(.003,.01)
 else:
  if not a.confirm_fresh:raise SystemExit('fresh requires --confirm-fresh')
  sel=ROOT/'DEV_SELECTION.json'
  if not sel.exists():raise SystemExit('development selection missing')
  seeds=(37210,37211,37212);lrs=(float(json.loads(sel.read_text())['selected_learning_rate']),)
 rows=[];scores={lr:[] for lr in lrs}
 for seed in seeds:
  for lr in lrs:
   rr=runworld(seed,lr,a.phase);rows+=rr
   if a.phase=='development':scores[lr]+=[float(x['nll']) for x in rr]
   print(json.dumps({'phase':a.phase,'world':seed,'lr':lr,'rows':len(rr)},sort_keys=True),flush=True)
 out=OUT/('development.csv' if a.phase=='development' else 'fresh.csv');out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w',newline='') as f:q=csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n');q.writeheader();q.writerows(rows)
 core=ROOT/'RESULTS_CORE.csv';old=list(csv.DictReader(core.open()))
 with core.open('w',newline='') as f:q=csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n');q.writeheader();q.writerows(old+rows)
 if a.phase=='development':
  sel=min(scores,key=lambda x:sum(scores[x])/len(scores[x]));(ROOT/'DEV_SELECTION.json').write_text(json.dumps({'experiment_id':'MA-372','selected_learning_rate':sel,'mean_validation_nll_by_lr':{str(k):sum(v)/len(v) for k,v in scores.items()},'development_worlds':list(seeds),'fresh_worlds_locked':[37210,37211,37212]},indent=2)+'\n')
if __name__=='__main__':main()
