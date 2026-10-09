#!/usr/bin/env python3
import json,hashlib,io,time,csv
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'artifacts';P=A/'payloads';D=16;R=4;STEP=20
DEV=[44800,44801];FRESH=[44830,44831,44832]
def world_basis(w):
 g=torch.Generator().manual_seed(w+889100);q,_=torch.linalg.qr(torch.randn(D,R,generator=g));return q
def task(w,tid,seed):
 g=torch.Generator().manual_seed(w*1000003+tid*991+seed*19);z=torch.randn(R,generator=g)*.5;target=world_basis(w)@z;x=torch.randn(64,D,generator=g);y=x@target;qx=torch.randn(256,D,generator=g);return target,x,y,qx,qx@target
def fit(w,tid,seed):
 target,x,y,qx,qy=task(w,tid,seed);theta=nn.Parameter(torch.zeros(D));opt=torch.optim.Adam([theta],lr=.05)
 for _ in range(STEP):opt.zero_grad();((x@theta-y).square().mean()).backward();opt.step()
 st=opt.state[theta];return {'theta':theta.detach(),'m':st['exp_avg'].detach(),'v':st['exp_avg_sq'].detach(),'target':target,'x':x,'y':y,'qx':qx,'qy':qy}
def continue_task(state,m,v):
 theta=nn.Parameter(state['theta'].clone());opt=torch.optim.Adam([theta],lr=.05);opt.state[theta]={'step':torch.tensor(float(STEP)),'exp_avg':m.clone(),'exp_avg_sq':v.clone(),'max_exp_avg_sq':torch.empty(0)}
 for _ in range(5):opt.zero_grad();((state['x']@theta-state['y']).square().mean()).backward();opt.step()
 with torch.no_grad():return (((state['qx']@theta-state['qy']).square().mean().sqrt())/(state['qy'].square().mean().sqrt()+1e-12)).item()
def main():
 A.mkdir(exist_ok=True);P.mkdir(exist_ok=True);dev=[]
 for w in DEV:
  for tid in range(100,132):dev.append(fit(w,tid,0))
 mat=torch.stack([torch.cat([s['m'],s['v']]) for s in dev]);mean=mat.mean(0);u,sv,vh=torch.linalg.svd(mat-mean,full_matrices=False);pca=vh[:R].T
 # Learned linear decoder initialized randomly, optimized to reconstruct the same development-state distribution.
 torch.manual_seed(44877);B=nn.Parameter(torch.randn(2*D,R)*.05);codes=nn.Parameter(torch.randn(len(dev),R)*.01);opt=torch.optim.Adam([B,codes],lr=.02)
 for _ in range(800):
  loss=((codes@B.T+mean-mat)**2).mean();opt.zero_grad();loss.backward();opt.step()
 B=B.detach();
 (A/'development_summary.json').write_text(json.dumps({'dev_tasks':len(dev),'rank':R,'pca_singular_values':sv[:R].tolist(),'learned_basis_train_loss':float(loss.detach()),'basis_train_updates':800},indent=2)+'\n')
 rows=[]; payloads={m:[] for m in ['exact','shared','pca','mirror']}
 for w in FRESH:
  states=[fit(w,tid,seed) for seed in [0,1,2] for tid in range(2000,2032)]
  testmat=torch.stack([torch.cat([s['m'],s['v']]) for s in states]);pca_code=(testmat-mean)@pca;mirror_code=torch.linalg.lstsq(B,testmat.T-mean[:,None]).solution.T
  recon={'exact':None,'shared':torch.zeros_like(testmat),'pca':mean+pca_code@pca.T,'mirror':mean+mirror_code@B.T}
  for method in recon:
   errs=[];codes_task=[]
   for i,s in enumerate(states):
    if method=='exact':m,v=s['m'],s['v']
    elif method=='shared':m,v=recon[method][i,:D],recon[method][i,D:]
    elif method=='pca':m,v=recon[method][i,:D],recon[method][i,D:].clamp_min(0)
    else:m,v=recon[method][i,:D],recon[method][i,D:].clamp_min(0)
    errs.append(continue_task(s,m,v));codes_task.append({'theta':s['theta'],'m':m,'v':v} if method=='exact' else {'theta':s['theta'],'code':(pca_code[i].contiguous() if method=='pca' else mirror_code[i].contiguous() if method=='mirror' else torch.empty(0))})
   payloads[method].append(codes_task)
   rows.append({'world':w,'method':method,'continuation_nrmse_mean':sum(errs)/len(errs),'tasks':len(errs),'state_reconstruction_nrmse':float((((recon[method] if recon[method] is not None else testmat)-testmat).square().mean().sqrt()/(testmat.square().mean().sqrt()+1e-12)) if method!='exact' else 0.)})
 # Whole-bank exact serialized payloads for N=32 fresh tasks/world.
 for method in payloads:
  for wi,w in enumerate(FRESH):
   tasks=payloads[method][wi]
   obj={'format':'ma448-compact-v1','method':method,'theta':torch.stack([t['theta'].clone() for t in tasks]),'optimizer_step':STEP}
   if method=='exact':obj.update(m=torch.stack([t['m'].clone() for t in tasks]),v=torch.stack([t['v'].clone() for t in tasks]))
   elif method in ('pca','mirror'):
    obj.update(codes=torch.stack([t['code'].clone() for t in tasks]),mean=mean.clone(),basis=(pca.clone() if method=='pca' else B.clone()))
   b=io.BytesIO();torch.save(obj,b);data=b.getvalue();path=P/f'{method}_{w}_N32.pt';path.write_bytes(data)
   row=next(x for x in rows if x['world']==w and x['method']==method);row.update(payload_bytes=len(data),bytes_per_task=len(data)/32,sha256=hashlib.sha256(data).hexdigest(),path=str(path.relative_to(ROOT.parents[2])))
 with open(A/'fresh_results.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
