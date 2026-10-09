#!/usr/bin/env python3
"""MA-613 sweep of private residual fraction in a shared function code bank."""
import argparse,hashlib,io,json,platform
from pathlib import Path
import torch
N,S=64,4
FREQ=torch.tensor([1.,2.,3.,4.]);FRACTIONS=[0.,.125,.25,.5,1.]

def make(seed):
    g=torch.Generator().manual_seed(seed);sig=torch.arange(N)%S;phase=torch.rand(N,generator=g)*2*torch.pi-torch.pi;private=torch.zeros(N,dtype=torch.bool)
    psi=torch.rand(N,generator=g)*2*torch.pi-torch.pi;amp=.35
    return sig,phase,private,psi,amp

def target(x,sig,phase,private,psi,amp):
    y=torch.sin(FREQ[sig,None]*x[None,:]+phase[:,None]);return y+private[:,None]*amp*torch.sin(5*x[None,:]+psi[:,None])
def full_codes(sig,phase,private,psi,amp):
    c=torch.zeros(N,10)
    for i in range(N):
        band=2*int(sig[i]);c[i,band]=phase[i].cos();c[i,band+1]=phase[i].sin()
        if private[i]:c[i,8]=amp*psi[i].cos();c[i,9]=amp*psi[i].sin()
    return c
def decode_mirror(x,sig,phase,res_idx,res_coeff):
    y=torch.sin(FREQ[sig,None]*x[None,:]+phase[:,None])
    if len(res_idx):y[res_idx]+=res_coeff[:,0,None]*torch.sin(5*x)[None,:]+res_coeff[:,1,None]*torch.cos(5*x)[None,:]
    return y
def decode_full(x,c):
    f=torch.arange(1,6,dtype=x.dtype);z=torch.stack((torch.sin(f[:,None]*x[None,:]),torch.cos(f[:,None]*x[None,:])),1).reshape(10,-1)
    return c@z
def pack(mode,seed,sig,phase,idx,coeff,codes):
    state={'mode':mode,'seed':seed,'sig':sig.to(torch.uint8),'phase':phase.float(),'residual_indices':idx.to(torch.uint8),'residual_coefficients':coeff.float().contiguous() if len(idx) else torch.empty(0,2),'full_codes':codes.float().contiguous() if mode=='full' else torch.empty(0,10),'metadata':{'n_functions':N,'frequencies':[1,2,3,4,5],'format':'MA613-v1'}}
    b=io.BytesIO();torch.save(state,b);return b.getvalue()
def run(seed):
    sig,phase,_,psi,amp=make(seed);x=torch.linspace(-torch.pi,torch.pi,512);out={};root=Path(__file__).resolve().parents[1]/'runs/dev_payloads';root.mkdir(parents=True,exist_ok=True)
    for frac in FRACTIONS:
        npriv=int(round(N*frac));private=torch.arange(N)<npriv;y=target(x,sig,phase,private,psi,amp);full=full_codes(sig,phase,private,psi,amp);fullpred=decode_full(x,full);fullblob=pack('full',seed,sig,phase,torch.empty(0,dtype=torch.long),torch.empty(0,2),full)
        fullpath=root/f'{seed}_p{npriv}_full.pt';fullpath.write_bytes(fullblob);fullnrmse=((fullpred-y)**2).mean().item()/y.var().item()
        out[f'{npriv}_full']={'private_fraction':frac,'residual_count':npriv,'method':'full','nrmse2':fullnrmse,'bytes':len(fullblob),'sha256':hashlib.sha256(fullblob).hexdigest(),'file':str(fullpath.relative_to(Path(__file__).resolve().parents[1])),'examples':N*len(x),'optimizer_updates':0,'compute_proxy':N*10}
        for keep_frac in ([0.,.5,1.] if npriv else [0.]):
            nkeep=int(round(npriv*keep_frac));idx=torch.arange(nkeep);rc=torch.stack((amp*psi[idx].cos(),amp*psi[idx].sin()),1) if nkeep else torch.empty(0,2);blob=pack('mirror',seed,sig,phase,idx,rc,full);yp=decode_mirror(x,sig,phase,idx,rc);err=((yp-y)**2).mean().item()/y.var().item();name=f'{npriv}_mirror_keep{nkeep}';path=root/f'{seed}_{name}.pt';path.write_bytes(blob)
            out[name]={'private_fraction':frac,'residual_count':nkeep,'private_recall':nkeep/max(1,npriv),'method':'mirror_sparse_residual','nrmse2':err,'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'file':str(path.relative_to(Path(__file__).resolve().parents[1])),'examples':N*len(x),'optimizer_updates':0,'compute_proxy':N*(3+(1 if nkeep else 0))}
    return out
def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=[61301,61302]);p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(1)
    d={'experiment_id':'MA-613','phase':'deterministic private-fraction sweep','python':platform.python_version(),'torch':torch.__version__,'device':'cpu','worlds':{str(s):run(s) for s in a.seeds}}
    o=Path(a.out);o.parent.mkdir(parents=True,exist_ok=True);o.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
if __name__=='__main__':main()
