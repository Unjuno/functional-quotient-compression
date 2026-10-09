#!/usr/bin/env python3
import csv,hashlib,io,json,math,statistics,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads'
DEV=[46100,46101];FRESH=[46110,46111,46112];SEEDS=[0,1,2];STEPS=[200,500,1000]
def rot(a):
 c,s=torch.cos(a),torch.sin(a);return torch.stack([c,-s,s,c],-1).reshape(*a.shape,2,2)
def contexts(world,seed,start,n=64):
 # Eight task identities x four layers x two adapter positions.
 g=torch.Generator().manual_seed(world*1000003+seed*997+start*53)
 task_attrs=torch.randn(8,2,generator=g)
 cs=[];ids=[]
 for t in range(8):
  for layer in range(4):
   for pos in range(2):
    cs.append(torch.tensor([task_attrs[t,0],task_attrs[t,1],(layer-1.5)/1.5, float(pos)*2-1]))
    ids.append(start+t*8+layer*2+pos)
 return torch.stack(cs)[:n],ids[:n]
def teacher(world,seed):
 g=torch.Generator().manual_seed(world*31337+seed*199)
 A0=torch.randn(4,generator=g)*.15
 H=torch.randn(4,4,generator=g)*.16
 return A0,H
def adapters(C,A0,H):return (A0+C@H).reshape(-1,2,2)
def observe(C,As,world,seed,start):
 x=[];y=[];q=[];qy=[]
 for i,A in enumerate(As):
  g=torch.Generator().manual_seed(world*99131+seed*71+start*17+i*23)
  xx=torch.randn(64,2,generator=g);qq=torch.randn(256,2,generator=torch.Generator().manual_seed(world*1217+seed*79+start*29+i*31))
  x.append(xx);y.append(xx@A.T);q.append(qq);qy.append(qq@A.T)
 return x,y,q,qy
def recover(x,y):return torch.linalg.solve(x.T@x+torch.eye(2)*1e-6,x.T@y).T
def normfit(C):
 mu=C.mean(0);sd=C.std(0).clamp_min(1e-5);return mu,sd
def aug(C,mu,sd):return torch.cat([(C-mu)/sd,torch.ones(len(C),1)],1)
def fullfit(C,A,mu,sd):
 X=aug(C,mu,sd);Y=A.reshape(len(A),4);W=torch.linalg.solve(X.T@X+torch.eye(5)*1e-6,X.T@Y);return W

def mirrorfit(C,A,mu,sd,steps):
 X=aug(C,mu,sd);W=torch.nn.Parameter(A.mean(0).clone()+torch.eye(2)*.05);G=torch.nn.Parameter(torch.randn(5,2)*.02);opt=torch.optim.Adam([W,G],lr=.025);begin=time.perf_counter()
 for _ in range(steps):
  opt.zero_grad();m=X@G;pred=rot(m[:,0])@W@rot(m[:,1]);loss=(pred-A).square().mean();loss.backward();opt.step()
 return {'W':W.detach(),'G':G.detach()},time.perf_counter()-begin

def rankfit(C,A,mu,sd,steps=500):
 X=aug(C,mu,sd);G=torch.nn.Parameter(torch.randn(5,2)*.02);B=torch.nn.Parameter(torch.randn(2,4)*.02);opt=torch.optim.Adam([G,B],lr=.025);begin=time.perf_counter()
 for _ in range(steps):
  opt.zero_grad();pred=(X@G)@B;loss=(pred-A.reshape(len(A),4)).square().mean();loss.backward();opt.step()
 return {'G':G.detach(),'B':B.detach()},time.perf_counter()-begin

def pred(method,C,state,mu,sd):
 X=aug(C,mu,sd)
 if method=='hyper':return (X@state).reshape(-1,2,2)
 if method=='mirror':
  m=X@state['G'];return rot(m[:,0])@state['W']@rot(m[:,1])
 z=X@state['G'];return (z@state['B']).reshape(-1,2,2)
def score(method,C,state,mu,sd,q,qy):
 As=state if method=='independent' else pred(method,C,state,mu,sd);errs=[];walls=[]
 for i in range(len(C)):
  t=time.perf_counter();out=q[i]@As[i].T;walls.append(time.perf_counter()-t);errs.append(float((out-qy[i]).square().mean().sqrt()/(qy[i].square().mean().sqrt()+1e-9)))
 return statistics.mean(errs),statistics.mean(walls)
def package(method,state,mu,sd,C,N):
 obj={'format':'ma461-v1','method':method,'N':N,'context_mean':mu,'context_std':sd,'contexts':C[:N]}
 if method=='independent':obj['adapter_matrices']=state[:N]
 elif method=='hyper':obj['generator']=state
 else:obj.update(state)
 b=io.BytesIO();torch.save(obj,b);return b.getvalue()
def one(world,seed,trainstart,teststart,steps,write=False):
 Ct,_=contexts(world,seed,trainstart);Ce,_=contexts(world,seed,teststart);A0,H=teacher(world,seed);At=adapters(Ct,A0,H);Ae=adapters(Ce,A0,H);xt,yt,_,_=observe(Ct,At,world,seed,trainstart);_,_,q,qy=observe(Ce,Ae,world,seed,teststart)
 Ahat=torch.stack([recover(xt[i],yt[i]) for i in range(len(Ct))]);mu,sd=normfit(Ct)
 Wf=fullfit(Ct,Ahat,mu,sd)
 torch.manual_seed(world+seed)
 sm,wm=mirrorfit(Ct,Ahat,mu,sd,steps)
 torch.manual_seed(world+seed+777)
 sr,wr=rankfit(Ct,Ahat,mu,sd)
 return Ct,Ce,Ae,q,qy,mu,sd,{'hyper':(Wf,0.),'mirror':(sm,wm),'rank2':(sr,wr),'independent':(Ae,0.)}
def main(phase):
 ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True)
 if phase=='development':
  scores=[]
  for steps in STEPS:
   vals=[]
   for w in DEV:
    for s in SEEDS:
     Ct,Ce,Ae,q,qy,mu,sd,models=one(w,s,100,5000,steps)
     for m in ['hyper','mirror','rank2']:
      e,_=score(m,Ce,models[m][0],mu,sd,q,qy);vals.append((m,e))
   scores.append({'steps':steps,'mirror_mean_nrmse':statistics.mean(e for m,e in vals if m=='mirror'),'hyper_mean_nrmse':statistics.mean(e for m,e in vals if m=='hyper'),'rank2_mean_nrmse':statistics.mean(e for m,e in vals if m=='rank2')})
  selected=min(scores,key=lambda r:r['mirror_mean_nrmse'])['steps'];(ART/'development_selection.json').write_text(json.dumps({'mirror_updates':selected,'scores':scores,'fresh_untouched':True},indent=2)+'\n');print(json.dumps({'selected':selected,'scores':scores}));return
 selected=json.loads((ART/'development_selection.json').read_text())['mirror_updates'];rows=[]
 for w in FRESH:
  for s in SEEDS:
   Ct,Ce,Ae,q,qy,mu,sd,models=one(w,s,2000,3000,selected)
   for method in ['hyper','mirror','rank2','independent']:
    state,fitwall=models[method];metric,querywall=score(method,Ce,state,mu,sd,q,qy)
    for N in [1,20,64]:
     b=package(method,state,mu,sd,Ce,N);path=PAY/f'{w}_{s}_{method}_N{N}.pt';path.write_bytes(b)
     params={'hyper':20,'mirror':14,'rank2':18,'independent':4*N}[method]
     rows.append({'world':w,'seed':s,'method':method,'n':N,'contexts':N,'nrmse_mean':metric,'payload_bytes':len(b),'bytes_per_context':len(b)/N,'parameter_count':params,'generator_MAC_per_context':({'hyper':20,'mirror':10,'rank2':18,'independent':0}[method]),'decoder_MAC_per_query':({'hyper':4,'mirror':16,'rank2':4,'independent':4}[method]),'optimizer_updates':(selected if method=='mirror' else 500 if method=='rank2' else 0),'fit_wall_seconds':fitwall,'query_wall_seconds_mean':querywall,'hash':hashlib.sha256(b).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 with open(ART/'fresh_runs.jsonl','w') as f:f.write(''.join(json.dumps(r)+'\n' for r in rows))
 with open(ROOT/'RESULTS_CORE.csv','w',newline='') as f:wri=csv.DictWriter(f,fieldnames=list(rows[0]));wri.writeheader();wri.writerows(rows)
 print(json.dumps({'phase':'fresh','rows':len(rows),'mirror_updates':selected}))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);main(p.parse_args().phase)
