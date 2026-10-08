from __future__ import annotations
import argparse,csv,hashlib,io,json,random,time
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from model import METHODS,ContextNet,Independent
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';UPDATES=1200;BATCH=128
FIELDS=['condition','world_or_seed','learning_rate','method','serialized_bytes','train_tokens_seen','optimizer_updates','base_MAC_per_token','generator_MAC_per_token','train_wall_time_s','inference_tokens_per_second','accuracy','nll','payload_sha256','roundtrip_max_abs_diff','status_note']
def seedall(s):random.seed(s);np.random.seed(s);torch.manual_seed(s)
def make_world(seed):
 g=torch.Generator().manual_seed(seed+43000);teacher=torch.Generator().manual_seed(seed+43500)
 w1=torch.randn(32,16,generator=teacher)*.45;b1=torch.randn(32,generator=teacher)*.15;head=torch.randn(4,32,generator=teacher)*.35;angles=torch.randn(4,4,generator=teacher)*.65
 data={}
 for name,n,salt in [('train',8192,101),('validation',2048,211),('test',4096,307)]:
  gg=torch.Generator().manual_seed(seed+salt);x=torch.randn(n,16,generator=gg);ctx=torch.arange(n)%4;ctx=ctx[torch.randperm(n,generator=gg)]
  h=F.relu(F.linear(x,w1,b1));pairs=h[:,:8].reshape(n,4,2);a,b=pairs[...,0],pairs[...,1];c=torch.cos(angles[ctx]);s=torch.sin(angles[ctx]);r=torch.stack((c*a-s*b,s*a+c*b),-1).reshape(n,8);h=torch.cat((r,h[:,8:]),1);logits=F.linear(h,head);y=logits.argmax(-1)
  data[name]=(x,ctx,y)
 return {'data':data,'teacher':{'w1':w1,'b1':b1,'head':head,'angles':angles}}
def train_method(w,seed,lr,method):
 seedall(seed+METHODS.index(method)*37)
 m=Independent() if method=='independent' else ContextNet(method);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);x,c,y=w['data']['train'];g=torch.Generator().manual_seed(seed+91);t=time.perf_counter();m.train()
 for _ in range(UPDATES):
  ids=torch.randint(len(y),(BATCH,),generator=g);loss=F.cross_entropy(m(x[ids],c[ids]),y[ids]);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 return m.eval(),time.perf_counter()-t
def pack(method,m,meta):
 obj={'method':method,'state':{k:v.detach().cpu() for k,v in m.state_dict().items()},'meta':meta};b=io.BytesIO();torch.save(obj,b);return b.getvalue()
def unpack(data):
 o=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False);m=Independent() if o['method']=='independent' else ContextNet(o['method']);m.load_state_dict(o['state']);m.eval();return m,o['method']
def evaluate(data,w,part):
 m,method=unpack(data);x,c,y=w['data'][part]
 with torch.no_grad():
  z=m(x,c);acc=float((z.argmax(-1)==y).float().mean());nll=float(F.cross_entropy(z,y));t=time.perf_counter()
  for _ in range(20):m(x[:256],c[:256])
  tps=20*min(len(y),256)/max(time.perf_counter()-t,1e-9)
 return acc,nll,tps
def runworld(seed,lr,phase):
 w=make_world(seed);rows=[]
 for method in METHODS:
  m,sec=train_method(w,seed+10000,lr,method);payload=pack(method,m,{'seed':seed,'lr':lr,'sequence_length':8});digest=hashlib.sha256(payload).hexdigest();path=OUT/'payloads'/phase/str(seed)/str(lr)/(method+'.pt');path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
  m2,mm=unpack(payload);x=w['data']['test'][0][:8];c=w['data']['test'][1][:8];rd=float((m(x,c)-m2(x,c)).abs().max());assert rd==0
  for part in ['validation' if phase=='development' else 'test']:
   acc,nll,tps=evaluate(payload,w,part);genmac={'shared':0,'mirror_gen':144,'film_gen':288,'static':0,'independent':0}[method];rows.append({'condition':phase,'world_or_seed':seed,'learning_rate':lr,'method':method,'serialized_bytes':len(payload),'train_tokens_seen':UPDATES*BATCH,'optimizer_updates':UPDATES,'base_MAC_per_token':640,'generator_MAC_per_token':genmac,'train_wall_time_s':round(sec,6),'inference_tokens_per_second':round(tps,3),'accuracy':f'{acc:.8f}','nll':f'{nll:.10f}','payload_sha256':digest,'roundtrip_max_abs_diff':rd,'status_note':f'{phase}/{seed}/{lr}/{method}; sequence length 8; complete inference state charged'})
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=('development','fresh'),required=True);p.add_argument('--confirm-fresh',action='store_true');a=p.parse_args();torch.set_num_threads(1)
 if a.phase=='development':seeds=(40300,40301);lrs=(.003,.01)
 else:
  if not a.confirm_fresh:raise SystemExit('fresh requires --confirm-fresh')
  sel=ROOT/'DEV_SELECTION.json';
  if not sel.exists():raise SystemExit('development selection missing')
  seeds=(40310,40311,40312);lrs=(float(json.loads(sel.read_text())['selected_learning_rate']),)
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
  sel=min(scores,key=lambda x:sum(scores[x])/len(scores[x]));(ROOT/'DEV_SELECTION.json').write_text(json.dumps({'experiment_id':'MA-403','selected_learning_rate':sel,'mean_validation_nll_by_lr':{str(k):sum(v)/len(v) for k,v in scores.items()},'development_worlds':list(seeds),'fresh_worlds_locked':[40310,40311,40312]},indent=2)+'\n')
if __name__=='__main__':main()
