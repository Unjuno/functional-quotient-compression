#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,platform,time
from pathlib import Path
import torch
N=32;NORB=24;NFREQ=[1.,2.,3.];GRID=1440;NS=16;NQ=512

def target(x,f,phase,private,psi):return torch.sin(f*x+phase)+(0.25*torch.sin(5*x+psi) if private else 0.)
def make(seed):
 g=torch.Generator().manual_seed(seed);freq=torch.tensor(NFREQ)[torch.randint(3,(N,),generator=g)];phase=torch.rand(N,generator=g)*2*torch.pi-torch.pi;private=torch.arange(N)>=NORB;psi=torch.rand(N,generator=g)*2*torch.pi-torch.pi
 xs=torch.rand(N,NS,generator=g)*2*torch.pi-torch.pi;xq=torch.linspace(-torch.pi,torch.pi,NQ);ys=torch.stack([target(xs[i],freq[i],phase[i],private[i].item(),psi[i]) for i in range(N)]);yq=torch.stack([target(xq,freq[i],phase[i],private[i].item(),psi[i]) for i in range(N)])
 return freq,phase,private,psi,xs,xq,ys,yq

def fit_mirror(freqs,x,y):
 grid=torch.linspace(-torch.pi,torch.pi,GRID);A=torch.stack((torch.sin(5*x),torch.cos(5*x)),1);pinv=torch.linalg.pinv(A);best=(1e30,None,None,None)
 for f in freqs:
  bases=torch.sin(f*x[:,None]+grid[None,:]);resid=y[:,None]-bases;coef=pinv@resid;recon=bases+A@coef;err=((recon-y[:,None])**2).mean(0);j=int(err.argmin())
  if float(err[j])<best[0]:best=(float(err[j]),float(f),grid[j],coef[:,j])
 return best[1],best[2],best[3],best[0],3*GRID*NS*6

def fit_coeff(freqs,x,y):
 best=(1e30,None,None,None)
 for f in freqs:
  A=torch.stack((torch.sin(f*x),torch.cos(f*x),torch.sin(5*x),torch.cos(5*x)),1);c=torch.linalg.lstsq(A,y).solution;err=((A@c-y)**2).mean().item()
  if err<best[0]:best=(err,float(f),c[:2],c[2:])
 return best[1],best[2],best[3],best[0],3*NS*32

def fit_full(x,y):
 fs=torch.arange(1,6,dtype=x.dtype);A=torch.cat((torch.sin(x[:,None]*fs),torch.cos(x[:,None]*fs)),1);c=torch.linalg.lstsq(A,y).solution;return c,((A@c-y)**2).mean().item(),10*NS*12

def pack(method,seed,values,private_ids,resid,full):
 d={'method':method,'seed':seed,'freqs':torch.tensor(NFREQ,dtype=torch.float16),'metadata':{'format':'MA618-v1','nfunc':N,'support':NS,'queries':NQ}}
 if method=='mirror':d['signatures']=values[0].to(torch.uint8);d['phases']=values[1].half()
 if method=='coeff':d['signatures']=values[0].to(torch.uint8);d['base_coeff']=values[1].half()
 if method in ('mirror','coeff'):d['private_indices']=private_ids.to(torch.uint8);d['private_residual']=resid.half()
 if method=='full':d['coefficients']=full.half()
 if method=='none':d['shared_only']=True
 b=io.BytesIO();torch.save(d,b);return b.getvalue()

def run(seed,phase_name):
 freq,phase,private,psi,xs,xq,ys,yq=make(seed);grid=torch.linspace(-torch.pi,torch.pi,GRID);mirror_vals=[];coef_vals=[];mir_res=[];coef_res=[];full=[];merr=[];cerr=[];ferr=[];mops=cops=fops=0;mf=cf=ff=0.;mq=cq=fq=nq=0.
 for i in range(N):
  t0=time.perf_counter();f,p,r,e,op=fit_mirror(NFREQ,xs[i],ys[i]);mf+=time.perf_counter()-t0;base=torch.sin(f*xq+p);A=torch.stack((torch.sin(5*xq),torch.cos(5*xq)),1);res=r;t0=time.perf_counter();pred=base+A@res;mq+=time.perf_counter()-t0;mirror_vals.append((f,p));mir_res.append(res);merr.append(((pred-yq[i])**2).mean().item()/float(yq[i].var(unbiased=False)+1e-12));mops+=op;mf+=0
  t0=time.perf_counter();f,c,r,e,op=fit_coeff(NFREQ,xs[i],ys[i]);cf+=time.perf_counter()-t0;base=torch.stack((torch.sin(f*xq),torch.cos(f*xq)),1)@c;A=torch.stack((torch.sin(5*xq),torch.cos(5*xq)),1);t0=time.perf_counter();pred=base+A@r;cq+=time.perf_counter()-t0;coef_vals.append((f,c));coef_res.append(r);cerr.append(((pred-yq[i])**2).mean().item()/float(yq[i].var(unbiased=False)+1e-12));cops+=op
  t0=time.perf_counter();co,ee,op=fit_full(xs[i],ys[i]);ff+=time.perf_counter()-t0;fs=torch.arange(1,6,dtype=xq.dtype);B=torch.cat((torch.sin(xq[:,None]*fs),torch.cos(xq[:,None]*fs)),1);t0=time.perf_counter();pred=B@co;fq+=time.perf_counter()-t0;full.append(co);ferr.append(((pred-yq[i])**2).mean().item()/float(yq[i].var(unbiased=False)+1e-12));fops+=op
 vals_m=(torch.tensor([NFREQ.index(v[0]) for v in mirror_vals]),torch.tensor([v[1] for v in mirror_vals]));vals_c=(torch.tensor([NFREQ.index(v[0]) for v in coef_vals]),torch.stack([v[1] for v in coef_vals]))
 root=Path(__file__).resolve().parents[1]/f'runs/{phase_name}_payloads';root.mkdir(parents=True,exist_ok=True);recs=[]
 methods=[('mirror',pack('mirror',seed,vals_m,torch.arange(NORB,N),torch.stack(mir_res[NORB:]),None),merr,mops,mf,mq),('coeff',pack('coeff',seed,vals_c,torch.arange(NORB,N),torch.stack(coef_res[NORB:]),None),cerr,cops,cf,cq),('full',pack('full',seed,None,torch.empty(0),torch.empty(0),torch.stack(full)),ferr,fops,ff,fq),('none',pack('none',seed,None,torch.empty(0),torch.empty(0),None),[float((yq[i]**2).mean()/(yq[i].var(unbiased=False)+1e-12)) for i in range(N)],0,0.,nq)]
 for name,blob,errs,ops,fit_wall,query_wall in methods:
  path=root/f'{seed}_{name}.pt';path.write_bytes(blob);recs.append({'method':name,'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'file':str(path.relative_to(root.parent)),'mean_nrmse2':sum(errs)/N,'on_orbit_mean_nrmse2':sum(errs[:NORB])/NORB,'private_mean_nrmse2':sum(errs[NORB:])/(N-NORB),'fit_compute_proxy':ops,'query_compute_proxy':N*5*NQ,'fit_wall_time_s':fit_wall,'query_wall_time_s':query_wall,'total_wall_time_s':fit_wall+query_wall,'examples':N*(NS+NQ),'optimizer_updates':0})
 return {'seed':seed,'phase':phase_name,'on_orbit':NORB,'private':N-NORB,'records':recs}
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['dev','fresh'],required=True);p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(1);seeds=[61801,61802] if a.phase=='dev' else [61811,61812,61813];ws=[run(s,a.phase) for s in seeds];d={'experiment_id':'MA-618','python':platform.python_version(),'torch':torch.__version__,'device':'cpu','worlds':ws};o=Path(a.out);o.parent.mkdir(parents=True,exist_ok=True);o.write_text(json.dumps(d,indent=2)+'\n')
 with (o.parent.parent/'RESULTS_CORE.csv').open('w',newline='') as f:
  wr=csv.writer(f);wr.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note'])
  for w in ws:
   for r in w['records']:wr.writerow([a.phase,w['seed'],r['method'],r['bytes'],r['examples'],0,r['fit_compute_proxy']+r['query_compute_proxy'],r['total_wall_time_s'],'mean_nrmse2',r['mean_nrmse2'],'private_mean_nrmse2',r['private_mean_nrmse2'],r['sha256']])
 print(json.dumps(d,indent=2))
if __name__=='__main__':main()
