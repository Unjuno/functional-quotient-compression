from __future__ import annotations
import argparse,csv,hashlib,io,json,random,time
from pathlib import Path
import numpy as np, sklearn, torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from model import WIDTHS,METHODS,NestedNet,Independent,macs,adapt_macs
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';UPDATES=700;BATCH=64;CODE_UPDATES=120
FIELDS=['condition','world_or_seed','learning_rate','method','width','serialized_bytes','train_examples_seen','optimizer_updates','active_MAC_proxy','adapt_MAC_proxy','train_wall_time_s','inference_examples_per_second','accuracy','nll','payload_sha256','roundtrip_max_abs_diff','status_note']
DATA_HASH='faf2d98f61250c1cdc0b114323133d3c58da04f5c5d28bb78e4bde3c936a75b1'
def seedall(s): random.seed(s);np.random.seed(s);torch.manual_seed(s)
def world(s):
 d=load_digits();h=hashlib.sha256(d.data.astype('float32').tobytes()+d.target.astype('int64').tobytes()).hexdigest()
 if h!=DATA_HASH or sklearn.__version__!='1.8.0': raise RuntimeError('data provenance mismatch')
 ids=np.arange(len(d.target));tr,rest=train_test_split(ids,train_size=.6,stratify=d.target,random_state=s);va,te=train_test_split(rest,test_size=.5,stratify=d.target[rest],random_state=s+1)
 return {'x':torch.tensor(d.data.astype('float32')/16),'y':torch.tensor(d.target.astype('int64')),'tr':torch.tensor(tr),'va':torch.tensor(va),'te':torch.tensor(te)}
def batch(w,n,g):
 ix=w['tr'][torch.randint(len(w['tr']),(n,),generator=g)];return w['x'][ix],w['y'][ix]
def fitbase(w,seed,lr):
 seedall(seed);m=NestedNet();o=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);g=torch.Generator().manual_seed(seed+100)
 t=time.perf_counter();m.train()
 for _ in range(UPDATES):
  x,y=batch(w,BATCH,g); widths=(64,16,WIDTHS[int(torch.randint(1,3,(),generator=g))]);teacher=m(x,64);loss=F.cross_entropy(teacher,y)
  for width in (16,widths[2]):
   z=m(x,width);loss=loss+F.cross_entropy(z,y)+.25*F.kl_div(F.log_softmax(z/2,-1),F.softmax(teacher.detach()/2,-1),reduction='batchmean')*4
  o.zero_grad(set_to_none=True);loss.backward();o.step()
 m.eval();return m,time.perf_counter()-t,UPDATES*BATCH*3,sum(macs(x) for x in (64,16,32))*UPDATES*BATCH

def fit_codes(m,w,seed,method):
 out={};elapsed=0.;
 for wi,width in enumerate(WIDTHS):
  if method=='mirror': c={f'l{i}':torch.nn.Parameter(torch.zeros(4)) for i in (1,2)}
  elif method=='film': c={f'l{i}':torch.nn.Parameter(torch.ones(4)) for i in (1,2)}
  else: c={'a':torch.nn.Parameter(torch.randn(width,1)*.01),'b':torch.nn.Parameter(torch.zeros(1,10))}
  o=torch.optim.Adam(c.values(),lr=.01);g=torch.Generator().manual_seed(seed+500+wi);ix=w['tr'];xall=w['x'][ix];yall=w['y'][ix];t=time.perf_counter()
  for p in m.parameters():p.requires_grad_(False)
  for _ in range(CODE_UPDATES):
   ids=torch.randint(len(yall),(BATCH,),generator=g);x,y=xall[ids],yall[ids]
   if method in ('mirror','film'): z=m(x,width,method,c)
   else:
    h=m.hidden(x,width);z=m(x,width)+(h@c['a'])@c['b']
   loss=F.cross_entropy(z,y);o.zero_grad(set_to_none=True);loss.backward();o.step()
  elapsed+=time.perf_counter()-t
  for p in m.parameters():p.requires_grad_(True)
  out[width]={k:v.detach().clone() for k,v in c.items()}
 return out,elapsed,CODE_UPDATES*BATCH*len(WIDTHS)
def train_ind(w,seed,lr):
 out={};elapsed=0
 for i,width in enumerate(WIDTHS):
  seedall(seed+900+i);m=Independent(width);o=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);g=torch.Generator().manual_seed(seed+1200+i);t=time.perf_counter()
  for _ in range(UPDATES):
   x,y=batch(w,BATCH,g);loss=F.cross_entropy(m(x),y);o.zero_grad(set_to_none=True);loss.backward();o.step()
  elapsed+=time.perf_counter()-t;m.eval();out[width]=m
 return out,elapsed,UPDATES*BATCH*3,sum(macs(x) for x in WIDTHS)*UPDATES*BATCH

def pred(m,c,ind,method,width,x):
 if method=='independent':return ind[width](x)
 if method in ('mirror','film'):return m(x,width,method,c[width])
 h=m.hidden(x,width);z=m(x,width)
 if method=='lora':z=z+(h@c[width]['a'])@c[width]['b']
 return z
def pack(method,m,c,ind,meta):
 obj={'method':method,'meta':meta,'shared':None if m is None else {k:v.detach().cpu() for k,v in m.state_dict().items()},'codes':{str(w):{k:v.detach().cpu() for k,v in c[w].items()} for w in c},'ind':None if ind is None else {str(w):{k:v.detach().cpu() for k,v in z.state_dict().items()} for w,z in ind.items()}}
 b=io.BytesIO();torch.save(obj,b);payload=b.getvalue();return payload,torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
def reload_bundle(b):
 method=b['method'];m=None;ind={};c={int(w):z for w,z in b['codes'].items()}
 if method!='independent':m=NestedNet();m.load_state_dict(b['shared']);m.eval()
 else:
  for w,s in b['ind'].items():z=Independent(int(w));z.load_state_dict(s);z.eval();ind[int(w)]=z
 return m,c,ind
def evaluate(payload,w,part,width):
 b=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False);m,c,ind=reload_bundle(b);method=b['method'];ix=w[part];x,y=w['x'][ix],w['y'][ix]
 with torch.no_grad():
  t=time.perf_counter()
  for _ in range(20):pred(m,c,ind,method,width,x[:256])
  tps=20*min(256,len(x))/max(time.perf_counter()-t,1e-9);z=pred(m,c,ind,method,width,x);return float((z.argmax(-1)==y).float().mean()),float(F.cross_entropy(z,y)),tps

def runworld(seed,lr,condition):
 w=world(seed);s=seed+20000;m,mt,mv,mm=fitbase(w,s,lr);codes={};ct={};
 for method in ('mirror','film','lora'):codes[method],ct[method],_=fit_codes(m,w,s,method)
 ind,it,iv,im=train_ind(w,s,lr);rows=[];meta={'widths':WIDTHS,'world':seed,'lr':lr,'dataset':'sklearn-digits'}
 for method in METHODS:
  if method=='nested':payload,_=pack(method,m,{},None,meta);trtime=mt;seen=mv;updates=UPDATES;active=mm;adapt=0
  elif method in codes:payload,_=pack(method,m,codes[method],None,meta);trtime=mt+ct[method];seen=mv+CODE_UPDATES*BATCH*3;updates=UPDATES+CODE_UPDATES*3;active=mm+sum(CODE_UPDATES*BATCH*(macs(width)+adapt_macs(method,width)) for width in WIDTHS);adapt=active-mm
  else:payload,_=pack(method,None,{},ind,meta);trtime=it;seen=iv;updates=UPDATES*3;active=im;adapt=0
  digest=hashlib.sha256(payload).hexdigest();path=OUT/'payloads'/condition/str(seed)/str(lr)/(method+'.pt');path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
  # Exact functional parity after deserialize on fixed diagnostic vectors.
  rb=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False);m2,c2,i2=reload_bundle(rb);diag=torch.linspace(0,1,5*64).reshape(5,64);rd=max(float((pred(m,codes.get(method,{}),ind,method,x,diag)-pred(m2,c2,i2,method,x,diag)).abs().max()) for x in WIDTHS)
  if rd!=0:raise RuntimeError('nonexact serialization roundtrip')
  for width in WIDTHS:
   acc,nll,tps=evaluate(payload,w,'va' if condition=='development' else 'te',width);rows.append({'condition':condition,'world_or_seed':seed,'learning_rate':lr,'method':method,'width':width,'serialized_bytes':len(payload),'train_examples_seen':seen,'optimizer_updates':updates,'active_MAC_proxy':active,'adapt_MAC_proxy':adapt,'train_wall_time_s':round(trtime,6),'inference_examples_per_second':round(tps,3),'accuracy':f'{acc:.8f}','nll':f'{nll:.10f}','payload_sha256':digest,'roundtrip_max_abs_diff':rd,'status_note':f'{condition}/{seed}/{lr}/{method}; all-width bank charged'})
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',required=True,choices=('development','fresh'));p.add_argument('--confirm-fresh',action='store_true');a=p.parse_args();torch.set_num_threads(1)
 if a.phase=='development':seeds=(37100,37101);lrs=(.003,.01)
 else:
  if not a.confirm_fresh:raise SystemExit('fresh requires --confirm-fresh')
  q=ROOT/'DEV_SELECTION.json'
  if not q.exists():raise SystemExit('freeze development selection first')
  lr=float(json.loads(q.read_text())['selected_learning_rate']);seeds=(37110,37111,37112);lrs=(lr,)
 rows=[];scores={lr:[] for lr in lrs}
 for s in seeds:
  for lr in lrs:
   rr=runworld(s,lr,a.phase);rows+=rr
   if a.phase=='development':scores[lr].extend(float(x['nll']) for x in rr)
   print(json.dumps({'phase':a.phase,'world':s,'lr':lr,'rows':len(rr)},sort_keys=True),flush=True)
 out=OUT/('development.csv' if a.phase=='development' else 'fresh.csv');out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w',newline='') as f:q=csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n');q.writeheader();q.writerows(rows)
 core=ROOT/'RESULTS_CORE.csv';old=list(csv.DictReader(core.open()));
 with core.open('w',newline='') as f:q=csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n');q.writeheader();q.writerows(old+rows)
 if a.phase=='development':
  sel=min(scores,key=lambda x:sum(scores[x])/len(scores[x]));(ROOT/'DEV_SELECTION.json').write_text(json.dumps({'experiment_id':'MA-371','selected_learning_rate':sel,'mean_validation_nll_by_lr':{str(k):sum(v)/len(v) for k,v in scores.items()},'development_worlds':list(seeds),'fresh_worlds_locked':[37110,37111,37112]},indent=2)+'\n')
if __name__=='__main__':main()
