from __future__ import annotations
import argparse,csv,hashlib,io,json,random,time
from pathlib import Path
import numpy as np,sklearn,torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from model import Net,Bank,METHODS,transform
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';UPDATES=600;CODE_UPDATES=100;BATCH=64;HASH='faf2d98f61250c1cdc0b114323133d3c58da04f5c5d28bb78e4bde3c936a75b1'
FIELDS=['condition','world_or_seed','learning_rate','method','context','serialized_bytes','train_examples_seen','optimizer_updates','active_MAC_proxy','adapt_MAC_proxy','train_wall_time_s','inference_examples_per_second','accuracy','nll','payload_sha256','roundtrip_max_abs_diff','status_note']
PERMS=[]
for i in range(4):PERMS.append(np.random.default_rng(401+i).permutation(64))
def seedall(s):random.seed(s);np.random.seed(s);torch.manual_seed(s)
def world(seed):
 d=load_digits();digest=hashlib.sha256(d.data.astype('float32').tobytes()+d.target.astype('int64').tobytes()).hexdigest()
 if digest!=HASH or sklearn.__version__!='1.8.0':raise RuntimeError('data provenance mismatch')
 ids=np.arange(len(d.target));tr,rest=train_test_split(ids,train_size=.6,stratify=d.target,random_state=seed);va,te=train_test_split(rest,test_size=.5,stratify=d.target[rest],random_state=seed+1)
 x=torch.tensor(d.data.astype('float32')/16);y=torch.tensor(d.target.astype('int64'));perms=[torch.tensor(p,dtype=torch.long) for p in PERMS]
 return {'x':x,'y':y,'tr':torch.tensor(tr),'va':torch.tensor(va),'te':torch.tensor(te),'perms':perms}
def get(w,part,ctx):ix=w[part];return w['x'][ix][:,w['perms'][ctx]],w['y'][ix]
def randbatch(w,g):
 ids=w['tr'][torch.randint(len(w['tr']),(BATCH,),generator=g)];ctx=int(torch.randint(4,(),generator=g));return w['x'][ids][:,w['perms'][ctx]],w['y'][ids],ctx
def train_shared(w,seed,lr):
 seedall(seed);m=Net();opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);g=torch.Generator().manual_seed(seed+11);t=time.perf_counter();m.train()
 for _ in range(UPDATES):
  x,y,c=randbatch(w,g);loss=F.cross_entropy(m(x),y);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 return m.eval(),time.perf_counter()-t
def fit_code(m,w,seed,method,ctx):
 if method=='mirror':code={'angles':torch.nn.Parameter(torch.zeros(4))}
 elif method=='film4':code={'scale':torch.nn.Parameter(torch.ones(4))}
 elif method=='film8':code={'scale':torch.nn.Parameter(torch.ones(4)),'bias':torch.nn.Parameter(torch.zeros(4))}
 else:code={'a':torch.nn.Parameter(torch.randn(64,1)*.01),'b':torch.nn.Parameter(torch.zeros(1,10))}
 opt=torch.optim.Adam(code.values(),lr=.01);g=torch.Generator().manual_seed(seed+100+ctx);xall,yall=get(w,'tr',ctx);t=time.perf_counter()
 for p in m.parameters():p.requires_grad_(False)
 for _ in range(CODE_UPDATES):
  ix=torch.randint(len(yall),(BATCH,),generator=g);x,y=xall[ix],yall[ix];h=m.hidden(x)
  if method=='rank1':z=m(x)+ (h@code['a'])@code['b']
  else:z=m.head(transform(h,method,code))
  loss=F.cross_entropy(z,y);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 for p in m.parameters():p.requires_grad_(True)
 return {k:v.detach().clone() for k,v in code.items()},time.perf_counter()-t

def train_independent(w,seed,lr):
 bank=Bank();tot=0.;
 for c in range(4):
  seedall(seed+500+c);m=bank.nets[c];opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);g=torch.Generator().manual_seed(seed+900+c);xall,yall=get(w,'tr',c);t=time.perf_counter()
  for _ in range(UPDATES):
   ix=torch.randint(len(yall),(BATCH,),generator=g);loss=F.cross_entropy(m(xall[ix]),yall[ix]);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
  tot+=time.perf_counter()-t;m.eval()
 return bank,tot

def predict(shared,codes,ind,method,ctx,x):
 if method=='independent':return ind.nets[ctx](x)
 if method=='shared':return shared(x)
 if method=='rank1':
  h=shared.hidden(x);return shared.head(h)+(h@codes[ctx]['a'])@codes[ctx]['b']
 return shared(x,method,codes[ctx])
def makepayload(method,shared,codes,ind,meta):
 obj={'method':method,'meta':meta,'shared':None if shared is None else shared.state_dict(),'codes':{str(c):{k:v.cpu() for k,v in d.items()} for c,d in codes.items()},'ind':None if ind is None else ind.state_dict(),'permutations':PERMS}
 b=io.BytesIO();torch.save(obj,b);return b.getvalue()
def loadpayload(data):
 o=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False);method=o['method'];shared=None;ind=None;codes={int(c):d for c,d in o['codes'].items()}
 if method=='independent':ind=Bank();ind.load_state_dict(o['ind']);ind.eval()
 else:shared=Net();shared.load_state_dict(o['shared']);shared.eval()
 return shared,codes,ind,method
def evalpayload(data,w,part,ctx):
 m,codes,ind,method=loadpayload(data);x,y=get(w,part,ctx)
 with torch.no_grad():
  z=predict(m,codes,ind,method,ctx,x);acc=float((z.argmax(-1)==y).float().mean());nll=float(F.cross_entropy(z,y));t=time.perf_counter()
  for _ in range(20):predict(m,codes,ind,method,ctx,x[:256])
  tps=20*min(256,len(x))/max(time.perf_counter()-t,1e-9)
 return acc,nll,tps
def runworld(seed,lr,phase):
 w=world(seed);s=seed+20000;m,basetime=train_shared(w,s,lr);codes_by={};ct={}
 for method in ('mirror','film4','film8','rank1'):
  codes_by[method]={};ct[method]=0
  for c in range(4):codes_by[method][c],dur=fit_code(m,w,s,method,c);ct[method]+=dur
 ind,indtime=train_independent(w,s,lr);rows=[]
 for method in METHODS:
  if method=='shared':shared=m;codes={};ib=None;trtime=basetime;updates=UPDATES;seen=UPDATES*BATCH;adapt=0
  elif method=='independent':shared=None;codes={};ib=ind;trtime=indtime;updates=4*UPDATES;seen=4*UPDATES*BATCH;adapt=0
  else:shared=m;codes=codes_by[method];ib=None;trtime=basetime+ct[method];updates=UPDATES+4*CODE_UPDATES;seen=UPDATES*BATCH+4*CODE_UPDATES*BATCH;adapt=4*CODE_UPDATES*BATCH*(24 if method=='mirror' else 8 if method=='film4' else 16 if method=='film8' else 74)
  payload=makepayload(method,shared,codes,ib,{'seed':seed,'lr':lr});digest=hashlib.sha256(payload).hexdigest();path=OUT/'payloads'/phase/str(seed)/str(lr)/(method+'.pt');path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
  # exact roundtrip over each context
  m2,c2,i2,mm=loadpayload(payload);rd=0.
  for c in range(4):
   xx=torch.linspace(0,1,3*64).reshape(3,64);rd=max(rd,float((predict(shared,codes,ib,method,c,xx)-predict(m2,c2,i2,mm,c,xx)).detach().abs().max()))
  if rd!=0:raise RuntimeError('payload function mismatch')
  for c in range(4):
   acc,nll,tps=evalpayload(payload,w,'va' if phase=='development' else 'te',c);rows.append({'condition':phase,'world_or_seed':seed,'learning_rate':lr,'method':method,'context':c,'serialized_bytes':len(payload),'train_examples_seen':seen,'optimizer_updates':updates,'active_MAC_proxy':64*64+64*10+(0 if method in ('shared','independent') else 24 if method=='mirror' else 8 if method=='film4' else 16 if method=='film8' else 74),'adapt_MAC_proxy':adapt,'train_wall_time_s':round(trtime,6),'inference_examples_per_second':round(tps,3),'accuracy':f'{acc:.8f}','nll':f'{nll:.10f}','payload_sha256':digest,'roundtrip_max_abs_diff':rd,'status_note':f'{phase}/{seed}/{lr}/{method}/ctx{c}; full bank bytes charged'})
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=('development','fresh'),required=True);p.add_argument('--confirm-fresh',action='store_true');a=p.parse_args();torch.set_num_threads(1)
 if a.phase=='development':seeds=(40100,40101);lrs=(.003,.01)
 else:
  if not a.confirm_fresh:raise SystemExit('fresh requires --confirm-fresh')
  sel=ROOT/'DEV_SELECTION.json';
  if not sel.exists():raise SystemExit('development selection missing')
  seeds=(40110,40111,40112);lrs=(float(json.loads(sel.read_text())['selected_learning_rate']),)
 rows=[];scores={lr:[] for lr in lrs}
 for seed in seeds:
  for lr in lrs:
   rr=runworld(seed,lr,a.phase);rows+=rr
   if a.phase=='development':scores[lr]+=[float(x['nll']) for x in rr]
   print(json.dumps({'phase':a.phase,'seed':seed,'lr':lr,'rows':len(rr)},sort_keys=True),flush=True)
 out=OUT/('development.csv' if a.phase=='development' else 'fresh.csv');out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w',newline='') as f:q=csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n');q.writeheader();q.writerows(rows)
 core=ROOT/'RESULTS_CORE.csv';old=list(csv.DictReader(core.open()))
 with core.open('w',newline='') as f:q=csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n');q.writeheader();q.writerows(old+rows)
 if a.phase=='development':
  sel=min(scores,key=lambda z:sum(scores[z])/len(scores[z]));(ROOT/'DEV_SELECTION.json').write_text(json.dumps({'experiment_id':'MA-401','selected_learning_rate':sel,'mean_validation_nll_by_lr':{str(k):sum(v)/len(v) for k,v in scores.items()},'development_worlds':list(seeds),'fresh_worlds_locked':[40110,40111,40112]},indent=2)+'\n')
if __name__=='__main__':main()
