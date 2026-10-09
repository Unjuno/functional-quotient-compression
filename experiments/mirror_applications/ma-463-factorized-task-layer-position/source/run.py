#!/usr/bin/env python3
import csv,hashlib,io,itertools,json,statistics,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads';DEV=[46300,46301];FRESH=[46320,46321,46322];SEEDS=[0,1,2];STEPS=[500,1500,3000]
COMB=list(itertools.product(range(8),range(4),range(3)))
def data(w,s,start):
 g=torch.Generator().manual_seed(w*1000003+s*997+start*53)
 u=torch.randn(8,generator=g)*.7;v=torch.randn(4,generator=g)*.7;z=torch.randn(3,generator=g)*.7
 A0=torch.randn(2,2,generator=g)*.15;B=torch.randn(2,2,generator=g)*.25
 targets=[];contexts=[];xs=[];ys=[];qs=[];qys=[];train=[];held=[]
 for i,(t,l,p) in enumerate(COMB):
  coeff=u[t]*v[l]*z[p];A=A0+coeff*B;contexts.append((t,l,p));targets.append(A)
  x=torch.randn(64,2,generator=torch.Generator().manual_seed(w*99131+s*71+start*17+i*23));q=torch.randn(256,2,generator=torch.Generator().manual_seed(w*1217+s*79+start*29+i*31))
  xs.append(x);ys.append(x@A.T);qs.append(q);qys.append(q@A.T)
  (held if (t+2*l+p)%4==0 else train).append(i)
 Ahat=torch.stack([torch.linalg.solve(xs[i].T@xs[i]+torch.eye(2)*1e-6,xs[i].T@ys[i]).T for i in range(96)])
 return contexts,Ahat,qs,qys,train,held

def input_feats(contexts):
 return torch.tensor([torch.nn.functional.one_hot(torch.tensor(t),8).float().tolist()+torch.nn.functional.one_hot(torch.tensor(l),4).float().tolist()+torch.nn.functional.one_hot(torch.tensor(p),3).float().tolist() for t,l,p in contexts])
def factorfit(contexts,A,steps):
 ut=torch.nn.Parameter(torch.randn(8)*.1);vl=torch.nn.Parameter(torch.randn(4)*.1);wp=torch.nn.Parameter(torch.randn(3)*.1);a0=torch.nn.Parameter(A.mean(0));b=torch.nn.Parameter(torch.randn(2,2)*.1);opt=torch.optim.Adam([ut,vl,wp,a0,b],lr=.025);t=time.perf_counter();ti=torch.tensor([c[0] for c in contexts]);li=torch.tensor([c[1] for c in contexts]);pi=torch.tensor([c[2] for c in contexts])
 for _ in range(steps):
  opt.zero_grad();pred=a0+(ut[ti]*vl[li]*wp[pi])[:,None,None]*b;loss=(pred-A).square().mean();loss.backward();opt.step()
 return {'task_codes':ut.detach(),'layer_codes':vl.detach(),'position_codes':wp.detach(),'A0':a0.detach(),'B':b.detach()},time.perf_counter()-t

def mlpfit(contexts,A,steps=1500):
 X=input_feats(contexts);net=torch.nn.Sequential(torch.nn.Linear(15,32),torch.nn.Tanh(),torch.nn.Linear(32,4));opt=torch.optim.Adam(net.parameters(),lr=.01);t=time.perf_counter()
 for _ in range(steps):
  opt.zero_grad();loss=(net(X)-A.reshape(len(A),4)).square().mean();loss.backward();opt.step()
 return {k:v.detach() for k,v in net.state_dict().items()},time.perf_counter()-t
def affinefit(contexts,A):
 X=input_feats(contexts);X=torch.cat([X,torch.ones(96,1)],1);return torch.linalg.solve(X.T@X+torch.eye(16)*1e-5,X.T@A.reshape(len(A),4))
def predict(m,contexts,st):
 if m in ('mirror','cp'):
  return torch.stack([st['A0']+(st['task_codes'][t]*st['layer_codes'][l]*st['position_codes'][p])*st['B'] for t,l,p in contexts])
 if m=='hyper':
  X=input_feats(contexts);net=torch.nn.Sequential(torch.nn.Linear(15,32),torch.nn.Tanh(),torch.nn.Linear(32,4));net.load_state_dict(st);return net(X).reshape(96,2,2)
 if m=='affine':
  X=torch.cat([input_feats(contexts),torch.ones(96,1)],1);return (X@st).reshape(96,2,2)
 return st

def evaluate(m,contexts,A,st,qs,qys,held):
 P=predict(m,contexts,st);errs=[];wall=[]
 for i in held:
  t=time.perf_counter();o=qs[i]@P[i].T;wall.append(time.perf_counter()-t);errs.append(float((o-qys[i]).square().mean().sqrt()/(qys[i].square().mean().sqrt()+1e-9)))
 return statistics.mean(errs),statistics.mean(wall)
def package(m,st,contexts,N):
 obj={'format':'ma463-v1','method':'factorized_product' if m in ('mirror','cp') else m,'N':N,'factor_indices':torch.tensor(contexts[:N],dtype=torch.uint8)}
 if m in ('mirror','cp'):obj.update(st)
 elif m=='hyper':obj.update(st)
 elif m=='affine':obj['decoder']=st
 else:obj['adapter_table']=st[:N]
 b=io.BytesIO();torch.save(obj,b);return b.getvalue()
def one(w,s,start,steps):
 c,A,q,qy,tr,held=data(w,s,start);idx=torch.tensor(tr);ctr=[c[i] for i in tr];Atr=A[idx];torch.manual_seed(w*1009+s*101+4242)
 # fit model on seen factor combinations only; context table indices remain fixed across decoder methods.
 fs,fw=factorfit([c[i] for i in tr],Atr,steps)
 # Mirror and ordinary CP are the identical product-factor parameterization; copy state for an exact control.
 cp={k:v.clone() for k,v in fs.items()}
 hs,hw=mlpfit([c[i] for i in tr],Atr)
 X=input_feats([c[i] for i in tr]);Xa=torch.cat([X,torch.ones(len(X),1)],1);aff=torch.linalg.solve(Xa.T@Xa+torch.eye(16)*1e-5,Xa.T@Atr.reshape(len(tr),4))
 # expand models' training-domain decoder to full ID range, enabling held-out combination evaluation.
 states={'mirror':fs,'cp':cp,'hyper':hs,'affine':aff,'independent':A}
 return c,A,q,qy,held,states,{'mirror':fw,'cp':fw,'hyper':hw,'affine':0.,'independent':0.}
def main(phase):
 ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True)
 if phase=='development':
  scores=[]
  for step in STEPS:
   ev=[]
   for w in DEV:
    for s in range(3):
     c,A,q,qy,h,states,_=one(w,s,100,step)
     for m in ['mirror','hyper']:
      e,_=evaluate(m,c,A,states[m],q,qy,h);ev.append((m,e))
   scores.append({'updates':step,'mirror_nrmse':statistics.mean(e for m,e in ev if m=='mirror'),'hyper_nrmse':statistics.mean(e for m,e in ev if m=='hyper')})
  chosen=min(scores,key=lambda r:r['mirror_nrmse'])['updates'];(ART/'development_selection.json').write_text(json.dumps({'mirror_updates':chosen,'scores':scores,'fresh_untouched':True},indent=2)+'\n');print(json.dumps({'selected':chosen,'scores':scores}));return
 steps=json.loads((ART/'development_selection.json').read_text())['mirror_updates'];rows=[]
 for w in FRESH:
  for s in range(3):
   c,A,q,qy,h,states,walls=one(w,s,3000,steps)
   for m in ['mirror','cp','hyper','affine','independent']:
    err,qwall=evaluate(m,c,A,states[m],q,qy,h)
    for N in [1,32,96]:
     b=package(m,states[m],c,N);path=PAY/f'{w}_{s}_{m}_N{N}.pt';path.write_bytes(b)
     rows.append({'world':w,'seed':s,'method':m,'n':N,'heldout_combos':len(h),'heldout_nrmse':err,'payload_bytes':len(b),'bytes_per_combo':len(b)/N,'decoder_MAC_per_combo':{'mirror':9,'cp':9,'hyper':128,'affine':64,'independent':4}[m],'optimizer_updates':steps if m in ('mirror','cp') else 1500 if m=='hyper' else 0,'fit_wall_seconds':walls[m],'query_wall_seconds_mean':qwall,'hash':hashlib.sha256(b).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 with open(ART/'fresh_runs.jsonl','w') as f:f.write(''.join(json.dumps(r)+'\n' for r in rows))
 with open(ROOT/'RESULTS_CORE.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':'fresh','rows':len(rows),'updates':steps}))
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=['development','fresh'],required=True);main(a.parse_args().phase)
