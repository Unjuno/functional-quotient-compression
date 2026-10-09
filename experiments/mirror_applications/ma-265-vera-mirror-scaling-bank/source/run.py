import csv,hashlib,io,json,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];D=32;R=8;N=8;DEV=[26500,26501];FRESH=[26510,26511,26512];SEEDS=[0,1,2];STEPS=300;torch.set_num_threads(2)
def basis():
 g=torch.Generator().manual_seed(265001);W0=torch.randn(D,D,generator=g)/D**.5;A=torch.randn(R,D,generator=g)/R**.5;B=torch.randn(D,R,generator=g)/R**.5;a0=torch.randn(R,generator=g)*.2;a1=torch.randn(R,generator=g)*.3;a2=torch.randn(R,generator=g)*.3;b0=torch.randn(R,generator=g)*.2;b1=torch.randn(R,generator=g)*.3;b2=torch.randn(R,generator=g)*.3;return W0,A,B,a0,a1,a2,b0,b1,b2
def tasks(w,s,stratum):
 W0,A,B,a0,a1,a2,b0,b1,b2=basis();g=torch.Generator().manual_seed(w*100003+s*7919+265);theta=(torch.rand(N,generator=g)*2-1)*3.0;aa=[];bb=[]
 for i in range(N):
  if stratum=='aligned':aa.append(a0+torch.cos(theta[i])*a1+torch.sin(theta[i])*a2);bb.append(b0+torch.cos(theta[i])*b1+torch.sin(theta[i])*b2)
  else:aa.append(torch.randn(R,generator=g)*.5);bb.append(torch.randn(R,generator=g)*.5)
 aa=torch.stack(aa);bb=torch.stack(bb);W=W0[None]+torch.stack([(B*bb[i][None,:])@(A*aa[i][:,None]) for i in range(N)]);return (W0,A,B,a0,a1,a2,b0,b1,b2,theta,aa,bb),W
def data(w,s,stratum,W):
 g=torch.Generator().manual_seed(w*100003+s*7919+333);x=torch.randn(N,320,D,generator=g);y=torch.einsum('ntd,ndh->nth',x,W);return x[:,:64],y[:,:64],x[:,64:],y[:,64:]
def weights(method,p,shared):
 W0,A,B,a0,a1,a2,b0,b1,b2=shared
 if method=='vera':aa=p['a'];bb=p['b'];return W0[None]+torch.stack([(B*bb[i][None,:])@(A*aa[i][:,None]) for i in range(N)])
 if method=='mirror':
  t=p['theta'];aa=torch.stack([a0+torch.cos(t[i])*a1+torch.sin(t[i])*a2 for i in range(N)]);bb=torch.stack([b0+torch.cos(t[i])*b1+torch.sin(t[i])*b2 for i in range(N)]);return W0[None]+torch.stack([(B*bb[i][None,:])@(A*aa[i][:,None]) for i in range(N)])
 if method=='generic':
  c=p['c'];s=p['s'];aa=a0+c[:,None]*a1+s[:,None]*a2;bb=b0+c[:,None]*b1+s[:,None]*b2;return W0[None]+torch.stack([(B*bb[i][None,:])@(A*aa[i][:,None]) for i in range(N)])
 return W0[None]+torch.stack([p['u'][i]@p['v'][i] for i in range(N)])
def init(method,seed):
 torch.manual_seed(seed)
 if method=='vera':return {'a':torch.nn.Parameter(torch.ones(N,R)*.1),'b':torch.nn.Parameter(torch.ones(N,R)*.1)}
 if method=='mirror':return {'theta':torch.nn.Parameter(torch.zeros(N))}
 if method=='generic':return {'c':torch.nn.Parameter(torch.zeros(N)),'s':torch.nn.Parameter(torch.zeros(N))}
 return {'u':torch.nn.Parameter(torch.randn(N,D,R)*.01),'v':torch.nn.Parameter(torch.randn(N,R,D)*.01)}
def fit(method,x,y,shared,seed):
 t=time.perf_counter()
 if method=='mirror':return fit_mirror_multistart(x,y,shared),time.perf_counter()-t
 p=init(method,seed);opt=torch.optim.Adam(list(p.values()),lr=.03)
 for _ in range(STEPS):
  opt.zero_grad();W=weights(method,p,shared);pred=torch.einsum('ntd,ndh->nth',x,W);loss=(pred-y).square().mean();loss.backward();opt.step()
 return {k:v.detach() for k,v in p.items()},time.perf_counter()-t
def fit_mirror_multistart(x,y,shared):
 W0,A,B,a0,a1,a2,b0,b1,b2=shared;angles=[]
 starts=torch.linspace(-3.0,3.0,9)
 for i in range(N):
  best=None;bestloss=float('inf')
  for start in starts:
   theta=torch.nn.Parameter(start.clone());opt=torch.optim.Adam([theta],lr=.03)
   for _ in range(200):
    opt.zero_grad();ca=torch.cos(theta);sa=torch.sin(theta);aa=a0+ca*a1+sa*a2;bb=b0+ca*b1+sa*b2;wi=W0+(B*bb[None,:])@(A*aa[:,None]);loss=(x[i]@wi-y[i]).square().mean();loss.backward();opt.step()
   value=float(loss.detach())
   if value<bestloss:bestloss=value;best=theta.detach().clone()
  angles.append(best)
 return {'theta':torch.stack(angles)}
def nrmse(a,b):return float((a-b).square().mean().sqrt()/b.square().mean().sqrt())
def ser(x):z=io.BytesIO();torch.save(x,z);return z.getvalue()
def run(phase):
 rows=[]
 for w in DEV if phase=='development' else FRESH:
  for seed in SEEDS:
   for stratum in ['aligned','independent']:
    pack,W=tasks(w,seed,stratum);W0,A,B,a0,a1,a2,b0,b1,b2,theta,aa,bb=pack;shared=(W0,A,B,a0,a1,a2,b0,b1,b2);xt,yt,xv,yv=data(w,seed,stratum,W)
    for m in ['vera','mirror','generic','independent_lora']:
     p,fitsec=fit(m,xt,yt,shared,seed+w);t=time.perf_counter();Wh=weights(m,p,shared);dec=time.perf_counter()-t;pred=torch.einsum('ntd,ndh->nth',xv,Wh);
     for _ in range(5):_=torch.einsum('ntd,ndh->nth',xv,Wh)
     app=[]
     for _ in range(20):
      ta=time.perf_counter();_=torch.einsum('ntd,ndh->nth',xv,Wh);app.append(time.perf_counter()-ta)
     applysec=sorted(app)[len(app)//2];err=nrmse(pred,yv)
     obj={'method':m,'base':W0,'shared_A':A if m!='independent_lora' else torch.empty(0),'shared_B':B if m!='independent_lora' else torch.empty(0),'scale_basis':[a0,a1,a2,b0,b1,b2] if m!='independent_lora' else [],'task_state':p,'metadata':{'N':N,'rank':R,'stratum':stratum,'world':w,'seed':seed}};blob=ser(obj);path=ROOT/'artifacts'/'payloads'/f'{w}_{seed}_{stratum}_{m}.pt';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob)
     rows.append({'world':w,'seed':seed,'stratum':stratum,'method':m,'nrmse':err,'payload_bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'path':str(path.relative_to(ROOT.parents[2])),'fit_seconds':fitsec,'decode_seconds':dec,'apply_seconds':applysec,'updates':STEPS,'tasks':N})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:wri=csv.DictWriter(f,fieldnames=rows[0]);wri.writeheader();wri.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
