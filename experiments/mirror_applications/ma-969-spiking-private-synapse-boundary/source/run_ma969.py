"""Real-event D1->D2 recurrent LIF adaptation frontier for MA-969."""
import argparse,csv,hashlib,io,json,math,pickle,struct,time,urllib.request,zlib,copy
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1];DATA=Path('/tmp/ma969_data');DATA.mkdir(parents=True,exist_ok=True)
URL='https://zenodo.org/api/records/20973757/files/datasets.zip/content';CD_OFF=715621894;CD_SIZE=10721
DOMAINS=('D1','D2');N_EVENTS=48;START_MS=80.0;N_NEURONS=8;N_IN=4;ALPHA=.90
FRESH_SEEDS=(96911,96913,96917);DATA_SPLIT_SEED=96900;K_VALUES=(1,2,4,8,16,32)
torch.set_num_threads(1)


def http_range(a,b):
 req=urllib.request.Request(URL,headers={'Range':f'bytes={a}-{b}','User-Agent':'MA969-worker/1.0'})
 with urllib.request.urlopen(req,timeout=60) as r:
  if r.status!=206:raise RuntimeError(f'expected range response; got {r.status}')
  return r.read()


def extract_members():
 if all((DATA/f'{d}_{n}.pkl').exists() for d in DOMAINS for n in ('events','labels')):return
 cd=http_range(CD_OFF,CD_OFF+CD_SIZE-1);entries={};pos=0
 while pos+46<=len(cd):
  if cd[pos:pos+4]!=b'PK\x01\x02':pos+=1;continue
  f=struct.unpack_from('<4s6H3I5H2I',cd,pos);method,crc,cs,us=f[4],f[7],f[8],f[9];nl,el,cl,off=f[10],f[11],f[12],f[16]
  name=cd[pos+46:pos+46+nl].decode();entries[name]=(method,crc,cs,us,off);pos+=46+nl+el+cl
 for d in DOMAINS:
  for n in ('events','labels'):
   name=f'datasets/ebtMNIST-{d}/Vernier/{n}.pkl';method,crc,cs,us,off=entries[name]
   lh=http_range(off,off+29);nl,el=struct.unpack_from('<HH',lh,26);start=off+30+nl+el;comp=http_range(start,start+cs-1)
   raw=zlib.decompress(comp,-15) if method==8 else comp
   if len(raw)!=us or zlib.crc32(raw)&0xffffffff!=crc:raise RuntimeError(f'bad CRC/size {name}')
   (DATA/f'{d}_{n}.pkl').write_bytes(raw)


def features_for(domain):
 ep=DATA/f'{domain}_events.pkl';lp=DATA/f'{domain}_labels.pkl'
 eb,lb=ep.read_bytes(),lp.read_bytes();events=pickle.loads(eb);labels=np.asarray(pickle.loads(lb),dtype=np.int64).reshape(-1)
 if len(events)!=len(labels):raise RuntimeError('event/label count mismatch')
 X=np.zeros((len(events),N_EVENTS,N_IN),np.float32)
 for i,e in enumerate(events):
  t=np.asarray(e['timestamp'],np.float64)*1e-3;xx=np.asarray(e['x'],np.float32);yy=np.asarray(e['y'],np.float32);pp=np.asarray(e['polarity'],np.float32)
  order=np.argsort(t,kind='stable');t,xx,yy,pp=t[order],xx[order],yy[order],pp[order]
  start=int(np.searchsorted(t,START_MS));start=min(start,max(0,len(t)-N_EVENTS));sl=slice(start,min(start+N_EVENTS,len(t)))
  n=len(t[sl]);
  if n<N_EVENTS: raise RuntimeError(f'{domain} trial {i} has only {n} events at selected window')
  tt=t[sl]
  # Event stream retains temporal order; normalize coordinates and relative time.
  p=pp[sl]
  if p.min()>=0 and p.max()<=1:p=p*2-1
  X[i,:,0]=(tt-START_MS)*.08
  X[i,:,1]=(xx[sl]-35.)/16.
  X[i,:,2]=(yy[sl]-26.)/16.
  X[i,:,3]=p
 meta={'domain':domain,'trials':len(events),'label_counts':{str(int(k)):int(v) for k,v in zip(*np.unique(labels,return_counts=True))},
       'events_sha256':hashlib.sha256(eb).hexdigest(),'labels_sha256':hashlib.sha256(lb).hexdigest(),
       'timestamp_scale':'microseconds to milliseconds, matching official loader','window_start_ms':START_MS,'window_events':N_EVENTS}
 return X,labels,meta


def split_indices(labels,seed):
 rng=np.random.default_rng(seed);tr=[];dv=[];te=[]
 for lab in sorted(np.unique(labels)):
  ix=np.flatnonzero(labels==lab);rng.shuffle(ix);n=len(ix);ntr=int(.6*n);ndv=int(.2*n)
  tr.extend(ix[:ntr]);dv.extend(ix[ntr:ntr+ndv]);te.extend(ix[ntr+ndv:])
 return np.asarray(sorted(tr)),np.asarray(sorted(dv)),np.asarray(sorted(te))


class Spike(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x):ctx.save_for_backward(x);return (x>=0).to(x.dtype)
 @staticmethod
 def backward(ctx,g):
  x,=ctx.saved_tensors;return g/(1.+x.abs()).pow(2)


class LIF(nn.Module):
 def __init__(self):
  super().__init__();self.w=nn.Parameter(torch.randn(N_NEURONS,N_IN)*.08);self.theta=nn.Parameter(torch.ones(N_NEURONS)*.45)
  self.readout=nn.Parameter(torch.randn(N_NEURONS)*.1);self.bias=nn.Parameter(torch.zeros(()))
 def forward(self,x):
  b=x.shape[0];v=x.new_zeros((b,N_NEURONS));counts=x.new_zeros((b,N_NEURONS))
  for t in range(N_EVENTS):
   v=ALPHA*v+x[:,t]@self.w.T;s=Spike.apply(v-self.theta);v=v-self.theta*s;counts=counts+s
  return counts@self.readout+self.bias,counts


def scores(model,X,y,indices,theta_override=None,w_override=None):
 x=torch.from_numpy(X[indices]);yt=torch.from_numpy((y[indices]>20).astype(np.int64))
 # evaluate with live model or adapter; all methods return logits and spike counts
 with torch.no_grad(): logits,counts=model(x,theta_override=theta_override,w_override=w_override) if isinstance(model,Adapt) else model(x)
 pred=(logits>=0).cpu().numpy();truth=yt.numpy();return float((pred==truth).mean()),float(counts.sum(-1).mean())


class Adapt(nn.Module):
 def __init__(self,base,kind='zero',k=0,mask=None,direction=None):
  super().__init__();self.base=base;self.kind=kind;self.k=k
  for p in self.base.parameters():p.requires_grad_(False)
  if kind=='direct':self.theta_d2=nn.Parameter(base.theta.detach().clone())
  elif kind in ('mirror','sparse'):self.m=nn.Parameter(torch.zeros(()));self.register_buffer('direction',torch.as_tensor(direction,dtype=torch.float32))
  if kind=='sparse':self.delta=nn.Parameter(torch.zeros(k));self.register_buffer('mask',torch.as_tensor(mask,dtype=torch.long))
 def effective(self,theta_override=None,w_override=None):
  w=self.base.w if w_override is None else w_override;theta=self.base.theta if theta_override is None else theta_override
  if self.kind=='direct' and theta_override is None:theta=self.theta_d2
  if self.kind in ('mirror','sparse') and theta_override is None:theta=self.base.theta+self.m*self.direction
  if self.kind=='sparse' and w_override is None:
   wf=w.reshape(-1).scatter(0,self.mask,self.delta);w=wf.reshape_as(w)
  return w,theta
 def forward(self,x,theta_override=None,w_override=None):
  w,theta=self.effective(theta_override,w_override);b=x.shape[0];v=x.new_zeros((b,N_NEURONS));counts=x.new_zeros((b,N_NEURONS))
  for t in range(N_EVENTS):
   v=ALPHA*v+x[:,t]@w.T;s=Spike.apply(v-theta);v=v-theta*s;counts=counts+s
  return counts@self.base.readout+self.base.bias,counts


def fit_base(X,y,train,dev,seed):
 torch.manual_seed(seed);base=LIF();opt=torch.optim.AdamW(base.parameters(),lr=.01,weight_decay=1e-4);xt=torch.from_numpy(X[train]);yt=torch.from_numpy((y[train]>20).astype(np.float32))
 best=-1;best_state=None;stale=0;updates=0;t0=time.perf_counter()
 for epoch in range(120):
  base.train();opt.zero_grad(set_to_none=True);logits,_=base(xt);loss=nn.functional.binary_cross_entropy_with_logits(logits,yt);loss.backward();opt.step();updates+=1
  base.eval();acc,_=scores(base,X,y,dev)
  if acc>best:best=acc;best_state={k:v.detach().clone() for k,v in base.state_dict().items()};stale=0
  else:
   stale+=1
   if stale>=12:break
 base.load_state_dict(best_state);return base,{'updates':updates,'wall_s':time.perf_counter()-t0,'best_dev':best}


def choose_mask(base,X,y,train,k):
 xt=torch.from_numpy(X[train]);yt=torch.from_numpy((y[train]>20).astype(np.float32));base.zero_grad(set_to_none=True)
 logits,_=base(xt);loss=nn.functional.binary_cross_entropy_with_logits(logits,yt);loss.backward()
 grad=base.w.grad.detach().abs().reshape(-1);return torch.topk(grad,k).indices.cpu().numpy()


def fit_adapt(base,kind,k,X,y,train,dev,seed,direction):
 torch.manual_seed(seed);mask=choose_mask(base,X,y,train,k) if kind=='sparse' else None
 model=Adapt(copy.deepcopy(base),kind,k,mask,direction);params=[p for p in model.parameters() if p.requires_grad]
 opt=torch.optim.AdamW(params,lr=.01,weight_decay=1e-4) if params else None
 xt=torch.from_numpy(X[train]);yt=torch.from_numpy((y[train]>20).astype(np.float32));best=-1;best_state=None;stale=0;updates=0;t0=time.perf_counter()
 if kind=='zero':return model,{'updates':0,'wall_s':0.,'best_dev':None,'mask':[] if mask is None else mask.tolist()}
 for epoch in range(100):
  model.train();opt.zero_grad(set_to_none=True);logits,_=model(xt);loss=nn.functional.binary_cross_entropy_with_logits(logits,yt);loss.backward();opt.step();updates+=1
  model.eval();acc,_=scores(model,X,y,dev)
  if acc>best:best=acc;best_state={k:v.detach().clone() for k,v in model.state_dict().items()};stale=0
  else:
   stale+=1
   if stale>=12:break
 model.load_state_dict(best_state);return model,{'updates':updates,'wall_s':time.perf_counter()-t0,'best_dev':best,'mask':[] if mask is None else mask.tolist()}


def fit_independent(X,y,train,dev,seed):
 return fit_base(X,y,train,dev,seed)


def serialize_joint(base,adapt,domain_meta):
 state={'shared_core':{k:v.detach().cpu() for k,v in base.state_dict().items()},'adaptation':{},'router':{'D1':0,'D2':1},'metadata':domain_meta}
 if adapt.kind=='direct':state['adaptation']={'kind':'direct','theta_d2':adapt.theta_d2.detach().cpu()}
 elif adapt.kind=='mirror':state['adaptation']={'kind':'mirror','m':adapt.m.detach().cpu(),'direction':adapt.direction.cpu()}
 elif adapt.kind=='sparse':state['adaptation']={'kind':'sparse','m':adapt.m.detach().cpu(),'direction':adapt.direction.cpu(),'indices':adapt.mask.cpu().to(torch.int16),'values':adapt.delta.detach().cpu()}
 else:state['adaptation']={'kind':'zero'}
 buf=io.BytesIO();torch.save(state,buf);return buf.getvalue()


def fresh_run(seed,dev_only=False):
 x1,y1,m1=features_for('D1');x2,y2,m2=features_for('D2')
 split1=split_indices(y1,DATA_SPLIT_SEED);split2=split_indices(y2,DATA_SPLIT_SEED+1)
 tr1,dv1,te1=split1;tr2,dv2,te2=split2
 base,fit1=fit_base(x1,y1,tr1,dv1,seed)
 eval1=dv1 if dev_only else te1;eval2=dv2 if dev_only else te2
 d1acc,d1sp=scores(base,x1,y1,eval1)
 # D1 core serialization and data provenance hashes are paid/recorded.
 d1_blob=serialize_joint(base,Adapt(copy.deepcopy(base),'zero'),{'base_domain':'D1','input_dim':N_IN,'neurons':N_NEURONS})
 direction=torch.ones(N_NEURONS)/math.sqrt(N_NEURONS)
 out=[];details=[]
 conditions=[('zero',0),('direct',0),('mirror',0)]+[('sparse',k) for k in K_VALUES]
 for ci,(kind,k) in enumerate(conditions):
  adapted,fitm=fit_adapt(base,kind,k,x2,y2,tr2,dv2,seed+100+ci,direction)
  eval2=dv2 if dev_only else te2;eval1=dv1 if dev_only else te1
  d2acc,d2sp=scores(adapted,x2,y2,eval2)
  blob=serialize_joint(base,adapted,{'base_domain':'D1','target_domain':'D2','input_dim':N_IN,'neurons':N_NEURONS,'kind':kind,'k':k})
  reloaded=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)
  # round-trip output from in-memory state equals loaded model state for trained adapter
  if kind=='direct':
   clone=Adapt(copy.deepcopy(base),'direct');clone.theta_d2.data.copy_(reloaded['adaptation']['theta_d2']);reloaded_d2,_=scores(clone,x2,y2,eval2)
  elif kind in ('mirror','sparse'):
   loaded_direction=reloaded['adaptation']['direction']
   loaded_mask=reloaded['adaptation']['indices'].to(torch.long).numpy() if kind=='sparse' else None
   clone=Adapt(copy.deepcopy(base),kind,k,loaded_mask,loaded_direction)
   clone.m.data.copy_(reloaded['adaptation']['m'])
   if kind=='sparse':clone.delta.data.copy_(reloaded['adaptation']['values'])
   reloaded_d2,_=scores(clone,x2,y2,eval2)
  else:reloaded_d2=d2acc
  if abs(reloaded_d2-d2acc)>1e-9:raise AssertionError('serialized replay mismatch')
  privbytes=len(blob)-len(d1_blob)
  active=48*(N_NEURONS*N_IN + (k if kind=='sparse' else 0))
  row={'seed':seed,'condition':kind,'k_private':k,'d1_payload_bytes':len(d1_blob),'joint_payload_bytes':len(blob),'private_bytes':privbytes,
       'd1_fresh_accuracy':d1acc,'d2_fresh_accuracy':d2acc,'spikes_per_sequence':d2sp,'active_mac_proxy':active,
       'optimizer_updates':fit1['updates']+fitm['updates'],'wall_time_s':fit1['wall_s']+fitm['wall_s'],
       'status_note':'real event D1->D2; whole-trial splits; one acquisition'}
  out.append(row);details.append({'row':row,'d2_adaptation':fitm,'mask':fitm.get('mask',[]),'payload_sha256':hashlib.sha256(blob).hexdigest()})
 # independent D2 model while retaining the D1 checkpoint in joint state.
 independent,fit2=fit_independent(x2,y2,tr2,dv2,seed+777);indacc,indsp=scores(independent,x2,y2,eval2)
 independent_state={'d1':{k:v.detach().cpu() for k,v in base.state_dict().items()},'d2':{k:v.detach().cpu() for k,v in independent.state_dict().items()},
                     'router':{'D1':0,'D2':1},'metadata':{'kind':'independent','input_dim':N_IN,'neurons':N_NEURONS}}
 buf=io.BytesIO();torch.save(independent_state,buf);ib=buf.getvalue()
 loaded=torch.load(io.BytesIO(ib),map_location='cpu',weights_only=False);loaded_ind=LIF();loaded_ind.load_state_dict(loaded['d2']);roundtrip,_=scores(loaded_ind,x2,y2,eval2)
 if abs(roundtrip-indacc)>1e-9:raise AssertionError('independent serialized replay mismatch')
 out.append({'seed':seed,'condition':'independent','k_private':N_NEURONS*N_IN,'d1_payload_bytes':len(d1_blob),'joint_payload_bytes':len(ib),'private_bytes':len(ib)-len(d1_blob),
             'd1_fresh_accuracy':d1acc,'d2_fresh_accuracy':indacc,'spikes_per_sequence':indsp,'active_mac_proxy':48*N_NEURONS*N_IN,
             'optimizer_updates':fit1['updates']+fit2['updates'],'wall_time_s':fit1['wall_s']+fit2['wall_s'],'status_note':'independent D2 SNN plus retained D1 SNN'})
 details.append({'row':out[-1],'payload_sha256':hashlib.sha256(ib).hexdigest(),'independent_dev':fit2['best_dev']})
 return out,{'seed':seed,'d1_data':m1,'d2_data':m2,'d1_split_counts':[len(z) for z in split1],'d2_split_counts':[len(z) for z in split2],
              'd1_fit':fit1,'results':details,'d1_evaluation_accuracy':d1acc,'evaluation_split':'development' if dev_only else 'fresh'}


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dev-only',action='store_true');args=ap.parse_args()
 extract_members()
 x1,y1,m1=features_for('D1');x2,y2,m2=features_for('D2')
 provenance={'dataset':'Zenodo Event-Based Tiny Images 20973757','d1':m1,'d2':m2,'license':'CC-BY-4.0; extracted data not committed'}
 (ROOT/'source'/'DATA_PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
 seeds=[96901] if args.dev_only else list(FRESH_SEEDS);allresults=[];field=next(csv.reader((ROOT/'RESULTS_CORE.csv').read_text().splitlines()))
 for seed in seeds:
  rows,detail=fresh_run(seed,args.dev_only);allresults.append(detail)
  if not args.dev_only:
   with (ROOT/'RESULTS_CORE.csv').open('a',newline='') as f:csv.DictWriter(f,fieldnames=field,lineterminator='\n').writerows(rows)
  print(json.dumps({'seed':seed,'dev_only':args.dev_only,'d1_eval_accuracy':detail['d1_evaluation_accuracy'],'rows':rows},sort_keys=True),flush=True)
 out='DEVELOPMENT.json' if args.dev_only else 'FRESH.json';(ROOT/'source'/out).write_text(json.dumps(allresults,indent=2)+'\n')

if __name__=='__main__':main()
