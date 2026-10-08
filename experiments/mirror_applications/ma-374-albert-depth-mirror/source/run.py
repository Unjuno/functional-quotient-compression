from __future__ import annotations
import argparse,csv,hashlib,io,json,random,time
from pathlib import Path
import numpy as np,sklearn,torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from model import METHODS,DepthNet
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';UPDATES=600;CODE_UPDATES=100;BATCH=64
FIELDS=['condition','world_or_seed','learning_rate','method','serialized_bytes','train_examples_seen','optimizer_updates','active_MAC_proxy','adapt_MAC_proxy','train_wall_time_s','inference_examples_per_second','accuracy','nll','payload_sha256','roundtrip_max_abs_diff','status_note'];HASH='faf2d98f61250c1cdc0b114323133d3c58da04f5c5d28bb78e4bde3c936a75b1'
def seedall(s):random.seed(s);np.random.seed(s);torch.manual_seed(s)
def world(s):
 d=load_digits();h=hashlib.sha256(d.data.astype('float32').tobytes()+d.target.astype('int64').tobytes()).hexdigest()
 if h!=HASH or sklearn.__version__!='1.8.0':raise RuntimeError('data provenance mismatch')
 ids=np.arange(len(d.target));tr,rest=train_test_split(ids,train_size=.6,stratify=d.target,random_state=s);va,te=train_test_split(rest,test_size=.5,stratify=d.target[rest],random_state=s+1)
 return {'x':torch.tensor(d.data.astype('float32')/16),'y':torch.tensor(d.target.astype('int64')),'tr':torch.tensor(tr),'va':torch.tensor(va),'te':torch.tensor(te)}
def batch(w,g):
 ix=w['tr'][torch.randint(len(w['tr']),(BATCH,),generator=g)];return w['x'][ix],w['y'][ix]
def fit(w,seed,lr,method):
 seedall(seed);m=DepthNet(method);o=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);g=torch.Generator().manual_seed(seed+99);t=time.perf_counter();m.train()
 for _ in range(UPDATES):
  x,y=batch(w,g);loss=F.cross_entropy(m(x),y);o.zero_grad(set_to_none=True);loss.backward();o.step()
 return m.eval(),time.perf_counter()-t

def fitcodes(m,w,seed,method):
 codes=[torch.nn.Parameter(torch.zeros(4) if method=='mirror' else torch.ones(4)) for _ in range(3)];opt=torch.optim.Adam(codes,lr=.01);g=torch.Generator().manual_seed(seed+777);ids=w['tr'];xall=w['x'][ids];yall=w['y'][ids]
 for p in m.parameters():p.requires_grad_(False)
 t=time.perf_counter()
 for _ in range(CODE_UPDATES):
  ix=torch.randint(len(yall),(BATCH,),generator=g);z=m(xall[ix],codes);loss=F.cross_entropy(z,yall[ix]);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 for p in m.parameters():p.requires_grad_(True)
 return [x.detach().clone() for x in codes],time.perf_counter()-t

def makepayload(method,m,codes,meta):
 obj={'method':method,'state':{k:v.detach().cpu() for k,v in m.state_dict().items()},'codes':None if codes is None else [c.cpu() for c in codes],'meta':meta};b=io.BytesIO();torch.save(obj,b);return b.getvalue()
def loadpayload(data):
 o=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False);m=DepthNet(o['method']);m.load_state_dict(o['state']);m.eval();return m,o['codes'],o['method']
def evaluate(data,w,part):
 m,c,method=loadpayload(data);ix=w[part];x,y=w['x'][ix],w['y'][ix]
 with torch.no_grad():
  z=m(x,c);acc=float((z.argmax(-1)==y).float().mean());nll=float(F.cross_entropy(z,y));t=time.perf_counter()
  for _ in range(20):m(x[:256],c)
  tps=20*min(len(x),256)/max(time.perf_counter()-t,1e-9)
 return acc,nll,tps

def runworld(seed,lr,phase):
 w=world(seed);rows=[]
 for method in METHODS:
  s=seed+10000;model,train_sec=fit(w,s,lr,method);codes=None;adapt=0.;updates=UPDATES;seen=UPDATES*BATCH
  if method in ('mirror','film'):
   codes,adapt=fitcodes(model,w,s,method);updates+=CODE_UPDATES;seen+=CODE_UPDATES*BATCH
  payload=makepayload(method,model,codes,{'world':seed,'lr':lr});path=OUT/'payloads'/phase/str(seed)/str(lr)/(method+'.pt');path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload);digest=hashlib.sha256(payload).hexdigest();m2,c2,me2=loadpayload(payload);diag=torch.linspace(0,1,6*64).reshape(6,64);rd=float((model(diag,codes)-m2(diag,c2)).detach().abs().max());assert rd==0
  for p in model.parameters():p.requires_grad_(True)
  for part in ['va' if phase=='development' else 'te']:
   acc,nll,tps=evaluate(payload,w,part);rows.append({'condition':phase,'world_or_seed':seed,'learning_rate':lr,'method':method,'serialized_bytes':len(payload),'train_examples_seen':seen,'optimizer_updates':updates,'active_MAC_proxy':3*(64*64+64*128+128*64)+640+ (72 if method=='mirror' else 24 if method=='film' else 0),'adapt_MAC_proxy':0 if not adapt else CODE_UPDATES*BATCH*3*3*64,'train_wall_time_s':round(train_sec+adapt,6),'inference_examples_per_second':round(tps,3),'accuracy':f'{acc:.8f}','nll':f'{nll:.10f}','payload_sha256':digest,'roundtrip_max_abs_diff':rd,'status_note':f'{phase}/{seed}/{lr}/{method}; full inference payload'})
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=('development','fresh'),required=True);p.add_argument('--confirm-fresh',action='store_true');a=p.parse_args();torch.set_num_threads(1)
 if a.phase=='development':seeds=(37400,37401);lrs=(.003,.01)
 else:
  if not a.confirm_fresh:raise SystemExit('fresh requires --confirm-fresh')
  sel=ROOT/'DEV_SELECTION.json';
  if not sel.exists():raise SystemExit('development selection missing')
  seeds=(37410,37411,37412);lrs=(float(json.loads(sel.read_text())['selected_learning_rate']),)
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
  sel=min(scores,key=lambda k:sum(scores[k])/len(scores[k]));(ROOT/'DEV_SELECTION.json').write_text(json.dumps({'experiment_id':'MA-374','selected_learning_rate':sel,'mean_validation_nll_by_lr':{str(k):sum(v)/len(v) for k,v in scores.items()},'development_worlds':list(seeds),'fresh_worlds_locked':[37410,37411,37412]},indent=2)+'\n')
if __name__=='__main__':main()
