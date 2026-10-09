#!/usr/bin/env python3
import csv,hashlib,io,itertools,json,statistics,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads';DEV=[46600,46601];FRESH=[46610,46611,46612];SEEDS=[0,1,2];UPDATES=1500
COMB=list(itertools.product(range(6),range(3),range(2)))
def phi(x):return torch.stack([x[:,0]*x[:,1],torch.ones(len(x)),torch.tanh(x[:,0])],1)
def make(w,s,start):
 g=torch.Generator().manual_seed(w*1000003+s*997+start*53);U=torch.randn(6,3,generator=g)*.5;V=torch.randn(3,3,generator=g)*.5;P=torch.randn(2,3,generator=g)*.5
 C=[];Gates=[];support=[];sy=[];query=[];qy=[];train=[];held=[]
 for i,(t,l,p) in enumerate(COMB):
  gate=U[t]*V[l]*P[p];C.append((t,l,p));Gates.append(gate)
  x=torch.randn(64,2,generator=torch.Generator().manual_seed(w*99131+s*71+start*17+i*23));q=torch.randn(256,2,generator=torch.Generator().manual_seed(w*1217+s*79+start*29+i*31));y=phi(x)@gate;support.append(x);sy.append(y);query.append(q);qy.append(phi(q)@gate)
  (held if (t+l+p)%4==0 else train).append(i)
 Ghat=[]
 for i in range(36):
  X=phi(support[i]);Ghat.append(torch.linalg.solve(X.T@X+torch.eye(3)*1e-5,X.T@sy[i]))
 return C,torch.stack(Gates),torch.stack(Ghat),support,sy,query,qy,train,held

def cpfit(C,Ghat,train):
 torch.manual_seed(4664321+len(train));U=torch.nn.Parameter(torch.randn(6,3)*.15);V=torch.nn.Parameter(torch.randn(3,3)*.15);P=torch.nn.Parameter(torch.randn(2,3)*.15);opt=torch.optim.Adam([U,V,P],lr=.025);t=time.perf_counter();ti=torch.tensor([C[i][0] for i in train]);li=torch.tensor([C[i][1] for i in train]);pi=torch.tensor([C[i][2] for i in train]);Y=Ghat[train]
 for _ in range(UPDATES):
  opt.zero_grad();pred=U[ti]*V[li]*P[pi];loss=(pred-Y).square().mean();loss.backward();opt.step()
 return {'task':U.detach(),'layer':V.detach(),'position':P.detach()},time.perf_counter()-t

def hyperfit(C,Ghat,train):
 X=onehot(C)[train];Y=Ghat[train];net=torch.nn.Sequential(torch.nn.Linear(11,32),torch.nn.Tanh(),torch.nn.Linear(32,3));opt=torch.optim.Adam(net.parameters(),lr=.01);t=time.perf_counter()
 for _ in range(UPDATES):opt.zero_grad();loss=(net(X)-Y).square().mean();loss.backward();opt.step()
 return {k:v.detach() for k,v in net.state_dict().items()},time.perf_counter()-t
def onehot(C):
 return torch.tensor([torch.nn.functional.one_hot(torch.tensor(t),6).float().tolist()+torch.nn.functional.one_hot(torch.tensor(l),3).float().tolist()+torch.nn.functional.one_hot(torch.tensor(p),2).float().tolist() for t,l,p in C])
def gatepred(m,C,st):
 if m in ('mirror','cp'):return torch.stack([st['task'][t]*st['layer'][l]*st['position'][p] for t,l,p in C])
 if m=='hyper':
  net=torch.nn.Sequential(torch.nn.Linear(11,32),torch.nn.Tanh(),torch.nn.Linear(32,3));net.load_state_dict(st);return net(onehot(C))
 if m=='unipelt':return st
 return None
def evaluate(m,C,G,st,q,qy,held,ablate=False):
 pred=gatepred(m,C,st);errs=[];walls=[];abl=[]
 for i in held:
  g=pred[i].clone()
  if ablate:g[ablate-1]=0
  t=time.perf_counter();out=phi(q[i])@g;walls.append(time.perf_counter()-t);errs.append(float((out-qy[i]).square().mean().sqrt()/(qy[i].square().mean().sqrt()+1e-9)))
 return statistics.mean(errs),statistics.mean(walls)
def package(m,st,C,N):
 obj={'format':'ma466-v1','method':'factorized_product' if m in ('mirror','cp') else m,'N':N,'combo_ids':torch.tensor(C[:N],dtype=torch.uint8),'component_definitions':['x1*x2','1','tanh(x1)']}
 if m in ('mirror','cp'):obj.update(st)
 elif m=='hyper':obj.update(st)
 elif m=='unipelt':obj['gate_table']=st[:N]
 b=io.BytesIO();torch.save(obj,b);return b.getvalue()
def run(phase):
 worlds=DEV if phase=='development' else FRESH;rows=[]
 if phase=='development':
  dev=[]
  for w in worlds:
   for s in range(3):
    C,G,Gh,x,y,q,qy,tr,h=make(w,s,100);st,wall=cpfit(C,Gh,tr);hs,hw=hyperfit(C,Gh,tr)
    dev.append({'world':w,'seed':s,'mirror':evaluate('mirror',C,G,st,q,qy,h)[0],'hyper':evaluate('hyper',C,G,hs,q,qy,h)[0]})
  (ART/'development_runs.json').write_text(json.dumps(dev,indent=2)+'\n');return
 for w in worlds:
  for s in range(3):
   C,G,Gh,x,y,q,qy,tr,h=make(w,s,3000);st,wall=cpfit(C,Gh,tr);hs,hw=hyperfit(C,Gh,tr);cp={k:v.clone() for k,v in st.items()};uni=G.clone()
   metrics={m:evaluate(m,C,G,z,q,qy,h) for m,z in [('mirror',st),('cp',cp),('hyper',hs),('unipelt',uni)]}
   abl={k:[evaluate('mirror',C,G,st,q,qy,h,ablate=j)[0] for j in (1,2,3)] for k in ['mirror']}
   for m,z in [('mirror',st),('cp',cp),('hyper',hs),('unipelt',uni)]:
    for N in [1,12,36]:
     b=package(m,z,C,N);path=PAY/f'{w}_{s}_{m}_N{N}.pt';path.write_bytes(b);err,qwall=metrics[m]
     rows.append({'world':w,'seed':s,'method':m,'n':N,'heldout_combos':len(h),'heldout_nrmse':err,'payload_bytes':len(b),'bytes_per_combo':len(b)/N,'decoder_MAC_per_combo':{'mirror':9,'cp':9,'hyper':384,'unipelt':3}[m],'optimizer_updates':UPDATES if m in ('mirror','cp','hyper') else 0,'fit_wall_seconds':wall if m in ('mirror','cp') else hw if m=='hyper' else 0.,'query_wall_seconds_mean':qwall,'ablated_component_nrmse_1':abl['mirror'][0],'ablated_component_nrmse_2':abl['mirror'][1],'ablated_component_nrmse_3':abl['mirror'][2],'hash':hashlib.sha256(b).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 with open(ART/'fresh_runs.jsonl','w') as f:f.write(''.join(json.dumps(r)+'\n' for r in rows))
 with open(ROOT/'RESULTS_CORE.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':'fresh','rows':len(rows)}))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
