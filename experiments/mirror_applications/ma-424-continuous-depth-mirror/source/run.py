#!/usr/bin/env python3
import argparse,hashlib,io,json,math,random,time
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';METHODS=['shared','mirror','scalar','rank1','independent'];DEV=[42400,42401];FRESH=[42410,42411,42412];SEEDS=[0,1,2];LRS=[.003,.01];M=4;D=2;UPDATES,BATCH=300,256
def fixseed(s):random.seed(s);torch.manual_seed(s);torch.set_num_threads(1)
def rot(a):
 c=a.cos();s=a.sin();return torch.stack([torch.stack([c,-s]),torch.stack([s,c])])
def world(w):
 g=torch.Generator().manual_seed(w);q=rot(torch.rand((),generator=g)*math.pi);base=q@torch.diag(torch.tensor([-.3,-1.4]))@q.T;angles=(torch.rand(M,generator=g)*1.4-.7);As=torch.stack([rot(a)@base@rot(a).T for a in angles]);x=torch.randn(M,1024,D,generator=g);dx=torch.einsum('mij,mbj->mbi',As,x)
 testx=torch.randn(M,128,D,generator=g);exact=torch.stack([torch.matrix_exp(As[i])@testx[i].T for i in range(M)]).transpose(1,2)
 return x,dx,testx,exact,As
class Model(nn.Module):
 def __init__(self,m):
  super().__init__();self.m=m
  if m=='independent':self.a=nn.Parameter(torch.randn(M,D,D)*.1)
  else:self.a=nn.Parameter(torch.randn(D,D)*.1)
  if m=='mirror':self.angles=nn.Parameter(torch.zeros(M))
  if m=='scalar':self.gate=nn.Parameter(torch.ones(M))
  if m=='rank1':self.u=nn.Parameter(torch.randn(M,D,1)*.01);self.v=nn.Parameter(torch.randn(M,1,D)*.01)
 def matrices(self):
  if self.m=='independent':return self.a
  if self.m=='mirror':return torch.stack([rot(self.angles[i])@self.a@rot(self.angles[i]).T for i in range(M)])
  if self.m=='scalar':return self.gate[:,None,None]*self.a
  if self.m=='rank1':return self.a[None]+self.u@self.v
  return self.a[None].expand(M,-1,-1)
 def forward(self,x,mode):return torch.bmm(self.matrices()[mode],x.unsqueeze(-1)).squeeze(-1)
def rk4(model,x0,mode,n):
 dt=1/n;x=x0
 for _ in range(n):
  k1=model(x,mode);k2=model(x+dt*k1/2,mode);k3=model(x+dt*k2/2,mode);k4=model(x+dt*k3,mode);x=x+dt*(k1+2*k2+2*k3+k4)/6
 return x
def serialize(m):
 b=io.BytesIO();torch.save({k:v.detach().cpu().contiguous() for k,v in m.state_dict().items()},b,_use_new_zipfile_serialization=True);return b.getvalue()
def run(m,w,s,lr,split):
 fixseed(w*100+s);x,dx,tx,exact,As=world(w);model=Model(m);opt=torch.optim.AdamW(model.parameters(),lr=lr);st=time.perf_counter();xx=x.reshape(-1,D);yy=dx.reshape(-1,D);mode=torch.arange(M).repeat_interleave(1024)
 for _ in range(UPDATES):
  ix=torch.randint(len(xx),(BATCH,));loss=((model(xx[ix],mode[ix])-yy[ix])**2).mean();opt.zero_grad();loss.backward();opt.step()
 wall=time.perf_counter()-st;txflat=tx.reshape(-1,D);modes=torch.arange(M)[:,None].expand(M,128).reshape(-1);pred8=rk4(model,txflat,modes,8).reshape(M,128,D);pred16=rk4(model,txflat,modes,16).reshape(M,128,D)
 err=lambda p:((p-exact).square().mean().sqrt()/(exact.square().mean().sqrt()+1e-9)).item();deriv=((model(xx,mode)-yy).square().mean().sqrt()/(yy.square().mean().sqrt()+1e-9)).item();eig=torch.linalg.eigvals(model.matrices()).abs();stiff=(eig.max(-1).values/eig.min(-1).values.clamp_min(1e-9)).tolist();raw=serialize(model);name=f'{split}_{w}_{s}_{m}_lr{lr:g}.pt';(OUT/'payloads'/name).write_bytes(raw);cl=Model(m);cl.load_state_dict(torch.load(io.BytesIO(raw),weights_only=True));rb=serialize(cl);assert raw==rb
 params=sum(p.numel() for p in model.parameters());return {'split':split,'world':w,'seed':s,'method':m,'lr':lr,'payload_file':name,'serialized_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'roundtrip_sha256':hashlib.sha256(rb).hexdigest(),'examples':UPDATES*BATCH,'updates':UPDATES,'parameter_count_diagnostic':params,'active_mac_proxy_per_state':2*D*D,'nfe_rk4_8':32,'nfe_rk4_16':64,'wall_time_s':round(wall,6),'derivative_normalized_rmse':deriv,'endpoint_normalized_rmse_rk4_8':err(pred8),'endpoint_normalized_rmse_rk4_16':err(pred16),'stiffness_ratios':stiff,'teacher_stiffness_ratio':4.6666665}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','fresh'],required=True);a=ap.parse_args();OUT.mkdir(exist_ok=True);(OUT/'payloads').mkdir(exist_ok=True);rows=[];sel={}
 if a.phase=='dev':
  for m in METHODS:
   sc={lr:[] for lr in LRS}
   for lr in LRS:
    for w in DEV:
     for s in SEEDS:r=run(m,w,s,lr,'development');rows.append(r);sc[lr].append(r['derivative_normalized_rmse'])
   sel[m]=min(LRS,key=lambda lr:sum(sc[lr])/len(sc[lr]))
  (OUT/'development_selection.json').write_text(json.dumps(sel,indent=2)+'\n');mode='w'
 else:
  sel=json.loads((OUT/'development_selection.json').read_text())
  for m in METHODS:
   for w in FRESH:
    for s in SEEDS:rows.append(run(m,w,s,sel[m],'fresh'))
  mode='a'
 with (OUT/'runs.jsonl').open(mode) as f:
  for r in rows:f.write(json.dumps(r,sort_keys=True)+'\n')
 print(json.dumps({'phase':a.phase,'rows':len(rows),'selection':sel},indent=2))
if __name__=='__main__':main()

