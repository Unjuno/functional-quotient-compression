#!/usr/bin/env python3
import csv,hashlib,io,json,math,statistics,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads';DEV=[46200,46201];FRESH=[46220,46221,46222];SEEDS=[0,1,2];STEPS=[300,800,1500]
def rot(a):
 c,s=torch.cos(a),torch.sin(a);return torch.stack([c,-s,s,c],-1).reshape(*a.shape,2,2)
def make(world,seed,start,n):
 g=torch.Generator().manual_seed(world*1000003+seed*997+start*53);ang=(torch.rand(n,2,generator=g)-.5)*2*math.pi
 C=torch.zeros(n,4);C[:,:2]=ang
 C[3*n//4:,2:]=(torch.rand(n-3*n//4,2,generator=g)-.5)*1.4
 W=torch.randn(2,2,generator=torch.Generator().manual_seed(world*31337+seed*199))*.3+torch.eye(2)*.7
 D1=torch.randn(2,2,generator=torch.Generator().manual_seed(world*4777+seed*71))*.25;D2=torch.randn(2,2,generator=torch.Generator().manual_seed(world*9973+seed*31))*.25
 A=rot(C[:,0])@W@rot(C[:,1])+C[:,2,None,None]*D1+C[:,3,None,None]*D2
 x=[];y=[];q=[];qy=[]
 for i in range(n):
  xx=torch.randn(64,2,generator=torch.Generator().manual_seed(world*99131+seed*71+start*17+i*23));qq=torch.randn(256,2,generator=torch.Generator().manual_seed(world*1217+seed*79+start*29+i*31));x.append(xx);y.append(xx@A[i].T);q.append(qq);qy.append(qq@A[i].T)
 return C,A,x,y,q,qy

def fourier_features(C):
 c1,c2=C[:,0],C[:,1];return torch.stack([torch.cos(c1)*torch.cos(c2),torch.cos(c1)*torch.sin(c2),torch.sin(c1)*torch.cos(c2),torch.sin(c1)*torch.sin(c2),C[:,2],C[:,3],torch.ones(len(C))],1)
def fourierfit(C,A):
 X=fourier_features(C);Y=A.reshape(len(A),4);return torch.linalg.solve(X.T@X+torch.eye(X.shape[1])*1e-6,X.T@Y)
def affine(C,A):
 X=torch.cat([C,torch.ones(len(C),1)],1);Y=A.reshape(len(C),4);return torch.linalg.solve(X.T@X+torch.eye(5)*1e-6,X.T@Y)
def rankfit(C,A,steps=800):
 X=torch.cat([C,torch.ones(len(C),1)],1);G=torch.nn.Parameter(torch.randn(5,2)*.05);B=torch.nn.Parameter(torch.randn(2,4)*.05);o=torch.optim.Adam([G,B],lr=.02);t=time.perf_counter()
 for _ in range(steps):
  o.zero_grad();loss=((X@G)@B-A.reshape(len(A),4)).square().mean();loss.backward();o.step()
 return {'G':G.detach(),'B':B.detach()},time.perf_counter()-t
def mirrorfit(C,A,steps=800):
 W=torch.nn.Parameter(A[:48].mean(0).clone());o=torch.optim.Adam([W],lr=.02);t=time.perf_counter()
 for _ in range(steps):
  o.zero_grad();p=rot(C[:,0])@W@rot(C[:,1]);loss=(p-A).square().mean();loss.backward();o.step()
 return {'W':W.detach()},time.perf_counter()-t
def mlpfit(C,A,steps):
 torch.manual_seed(1901);net=torch.nn.Sequential(torch.nn.Linear(4,16),torch.nn.Tanh(),torch.nn.Linear(16,4));o=torch.optim.Adam(net.parameters(),lr=.01);t=time.perf_counter();X=C;Y=A.reshape(len(A),4)
 for _ in range(steps):
  o.zero_grad();loss=(net(X)-Y).square().mean();loss.backward();o.step()
 return {k:v.detach() for k,v in net.state_dict().items()},time.perf_counter()-t
def predict(m,C,st):
 if m=='mirror':return rot(C[:,0])@st['W']@rot(C[:,1])
 if m=='affine':return (torch.cat([C,torch.ones(len(C),1)],1)@st).reshape(-1,2,2)
 if m=='rank2':return ((torch.cat([C,torch.ones(len(C),1)],1)@st['G'])@st['B']).reshape(-1,2,2)
 if m=='fourier':return (fourier_features(C)@st).reshape(-1,2,2)
 if m=='affine':return (torch.cat([C,torch.ones(len(C),1)],1)@st).reshape(-1,2,2)
 net=torch.nn.Sequential(torch.nn.Linear(4,16),torch.nn.Tanh(),torch.nn.Linear(16,4));net.load_state_dict(st);return net(C).reshape(-1,2,2)
def eval_model(m,C,A,st,q,qy):
 P=st if m=='private' else predict(m,C,st);e=[];wt=[]
 for i in range(len(C)):
  t=time.perf_counter();out=q[i]@P[i].T;wt.append(time.perf_counter()-t);e.append(float((out-qy[i]).square().mean().sqrt()/(qy[i].square().mean().sqrt()+1e-9)))
 return statistics.mean(e),statistics.mean(wt)
def package(m,st,C,N):
 obj={'format':'ma462-v1','method':m,'N':N,'fixed_embeddings':C[:N]}
 if m=='private':obj['private_adapters']=st[:N]
 elif m=='hyper':
  for k,v in st.items():obj[k]=v
 elif m in ('affine','fourier'):obj['decoder']=st
 else:obj.update(st)
 b=io.BytesIO();torch.save(obj,b);return b.getvalue()
def one(w,s,trainstart,teststart,steps):
 Ct,At,xt,yt,_,_=make(w,s,trainstart,64);Ce,Ae,_,_,q,qy=make(w,s,teststart,32)
 Ahat=torch.stack([torch.linalg.solve(xt[i].T@xt[i]+torch.eye(2)*1e-6,xt[i].T@yt[i]).T for i in range(64)])
 out={};out['hyper']=(mlpfit(Ct,Ahat,steps));out['mirror']=mirrorfit(Ct,Ahat);out['affine']=(affine(Ct,Ahat),0.);out['rank2']=rankfit(Ct,Ahat);out['private']=(Ae,0.)
 return Ct,Ahat,Ce,Ae,q,qy,out

def main(phase):
 ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True)
 if phase=='development':
  scores=[]
  for step in STEPS:
   vals=[]
   for w in DEV:
    for s in SEEDS:
     Ct,Ahat,Ce,Ae,q,qy,out=one(w,s,100,5000,step)
     for m in ['hyper','mirror']:
      e,_=eval_model(m,Ce,Ae,out[m][0],q,qy);vals.append((m,e))
   scores.append({'updates':step,'hyper_nrmse':statistics.mean(e for m,e in vals if m=='hyper'),'mirror_nrmse':statistics.mean(e for m,e in vals if m=='mirror')})
  best=min(scores,key=lambda r:r['hyper_nrmse'])['updates'];(ART/'development_selection.json').write_text(json.dumps({'hyper_updates':best,'scores':scores,'fresh_untouched':True},indent=2)+'\n');print(json.dumps({'selected':best,'scores':scores}));return
 steps=json.loads((ART/'development_selection.json').read_text())['hyper_updates'];rows=[]
 for w in FRESH:
  for s in SEEDS:
   Ct,Ahat,C,A,q,qy,out=one(w,s,4000,5000,steps)
   out['fourier']=(fourierfit(Ct,Ahat),0.)
   for m in ['hyper','mirror','affine','rank2','fourier','private']:
    st,fitwall=out[m];metric,qwall=eval_model(m,C,A,st,q,qy)
    for N in [1,16,32]:
     b=package(m,st,C,N);path=PAY/f'{w}_{s}_{m}_N{N}.pt';path.write_bytes(b)
     rows.append({'world':w,'seed':s,'method':m,'n':N,'contexts':N,'nrmse_mean':metric,'payload_bytes':len(b),'bytes_per_context':len(b)/N,'decoder_MAC_per_context':{'hyper':128,'mirror':16,'affine':20,'rank2':18,'fourier':28,'private':4}[m],'optimizer_updates':steps if m=='hyper' else 800 if m in ('mirror','rank2') else 0,'fit_wall_seconds':fitwall,'query_wall_seconds_mean':qwall,'hash':hashlib.sha256(b).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 with open(ART/'fresh_runs_A1.jsonl','w') as f:f.write(''.join(json.dumps(r)+'\n' for r in rows))
 with open(ROOT/'RESULTS_CORE_A1.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':'fresh_A1','rows':len(rows),'hyper_updates':steps}))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);main(p.parse_args().phase)
