#!/usr/bin/env python3
"""MA-608 direct fusion of shared adapter views before output projection."""
import argparse,hashlib,io,json,platform,time
from pathlib import Path
import torch
from torch import nn
D,O,R,K=32,32,6,8
PAIRS=[(i,j) for i in range(R) for j in range(i+1,R)]

def rotate(v,a):
    z=v.clone()
    for n,(i,j) in enumerate(PAIRS):
        c,s=a[n].cos(),a[n].sin();u,w=z[...,i].clone(),z[...,j].clone();z[...,i]=c*u-s*w;z[...,j]=s*u+c*w
    return z

def make(seed):
    g=torch.Generator().manual_seed(seed);a=torch.randn(R,D,generator=g)*.12;b=torch.randn(O,R,generator=g)*.12
    codes=torch.randn(K,len(PAIRS),generator=g)*.35
    eye=torch.eye(R); mats=torch.stack([rotate(eye,codes[k]) for k in range(K)])
    gate=nn.Linear(D,K)
    with torch.no_grad():gate.weight.copy_(torch.randn(K,D,generator=g)*.3);gate.bias.copy_(torch.randn(K,generator=g)*.1)
    return a,b,codes,mats,gate

def outputs(x,a,b,c):
    z=x@a.T
    return torch.stack([(z@c[t].T)@b.T for t in range(K)],1)

def fused_reference(x,a,b,c,gate,nonlinear=False):
    z=x@a.T;alpha=torch.softmax(gate(x),-1);ys=[]
    for t in range(K):
        v=z@c[t].T
        if nonlinear:v=torch.tanh(v)
        ys.append(v@b.T)
    return torch.einsum('nk,nko->no',alpha,torch.stack(ys,1))

def fused_full(x,afull,b,gate):
    alpha=torch.softmax(gate(x),-1)
    ys=torch.stack([(x@afull[t].T)@b.T for t in range(K)],1)
    return torch.einsum('nk,nko->no',alpha,ys)

def fused_direct(x,a,b,c,gate,nonlinear=False):
    alpha=torch.softmax(gate(x),-1);ce=torch.einsum('nk,kij->nij',alpha,c);z=x@a.T
    v=torch.einsum('nij,nj->ni',ce,z)
    if nonlinear:v=torch.tanh(v)
    return v@b.T

def payload(mode,a,b,codes,mats,gate):
    state={'gate':{k:v.detach().cpu().contiguous() for k,v in gate.state_dict().items()}}
    if mode=='full':
        # Independent factors A_t=C_t A, B_t=B reconstruct exactly the shared teacher maps.
        state['a_full']=torch.stack([mats[t]@a for t in range(K)]).contiguous();state['b_full']=b.unsqueeze(0).expand(K,-1,-1).contiguous()
    elif mode=='mirror':state.update({'a':a.contiguous(),'b':b.contiguous(),'angles':codes.contiguous()})
    else:state.update({'a':a.contiguous(),'b':b.contiguous(),'coeff_matrices':mats.contiguous()})
    bio=io.BytesIO();torch.save({'state':state,'mode':mode,'dims':[D,O,R,K],'format':'MA608-v1'},bio);return bio.getvalue()

def replay_payload(blob,x):
    d=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False);s=d['state'];g=nn.Linear(D,K);g.load_state_dict(s['gate'])
    if d['mode']=='full':
        alpha=torch.softmax(g(x),-1);ys=torch.stack([(x@s['a_full'][t].T)@s['b_full'][t].T for t in range(K)],1)
        return torch.einsum('nk,nko->no',alpha,ys)
    if d['mode']=='mirror':
        a,b,codes=s['a'],s['b'],s['angles'];mats=torch.stack([rotate(torch.eye(R),codes[t]) for t in range(K)])
    else:a,b,mats=s['a'],s['b'],s['coeff_matrices']
    return fused_direct(x,a,b,mats,g)

def flop_proxies():
    gate=2*D*K;proj=D*R;out=R*O;mix=K*R*R;nonlin=R
    return {'full_adapterfusion':gate+K*(D*R+R*O),'shared_adapterfusion':gate+proj+mix+K*out,'mirror_direct':gate+proj+mix+R*R+out,'generic_code_direct':gate+proj+mix+R*R+out}

def latency(fn,x,reps=80):
    fn(x);start=time.perf_counter()
    for _ in range(reps):fn(x)
    return (time.perf_counter()-start)*1000/reps

def run(seed):
    a,b,codes,mats,gate=make(seed);g=torch.Generator().manual_seed(seed+100);x=torch.randn(4096,D,generator=g)
    with torch.no_grad():
        alpha=torch.softmax(gate(x),-1);z=x@a.T
        # exact reference variants
        standard=fused_reference(x,a,b,mats,gate);direct=fused_direct(x,a,b,mats,gate)
        nonlinear_ref=fused_reference(x,a,b,mats,gate,True);nonlinear_direct=fused_direct(x,a,b,mats,gate,True)
        full_y=[]
        for t in range(K):full_y.append((x@(mats[t]@a).T)@b.T)
        full=torch.einsum('nk,nko->no',alpha,torch.stack(full_y,1))
        nrmse=((nonlinear_direct-nonlinear_ref)**2).mean().item()/nonlinear_ref.var().item()
        errs={'shared_direct_max_abs':(direct-standard).abs().max().item(),'full_vs_shared_max_abs':(full-standard).abs().max().item(),'nonlinear_direct_nrmse2':nrmse}
    root=Path(__file__).resolve().parents[1]/'runs/dev_payloads';root.mkdir(parents=True,exist_ok=True);sizes={};hashes={}
    replay={}
    for mode in ('full','mirror','matrix'):
        blob=payload(mode,a,b,codes,mats,gate);p=root/f'{seed}_{mode}.pt';p.write_bytes(blob);sizes[mode]=len(blob);hashes[mode]=hashlib.sha256(blob).hexdigest()
        with torch.no_grad():replay[mode]=(replay_payload(blob,x)-standard).abs().max().item()
    afull=torch.stack([mats[t]@a for t in range(K)])
    times={'full_adapterfusion_ms_per_4096':latency(lambda q:fused_full(q,afull,b,gate),x),
           'shared_adapterfusion_ms_per_4096':latency(lambda q:fused_reference(q,a,b,mats,gate),x),
           'mirror_direct_ms_per_4096':latency(lambda q:fused_direct(q,a,b,mats,gate),x),
           'nonlinear_reference_ms_per_4096':latency(lambda q:fused_reference(q,a,b,mats,gate,True),x),
           'nonlinear_direct_ms_per_4096':latency(lambda q:fused_direct(q,a,b,mats,gate,True),x)}
    return {'examples':4096,'optimizer_updates':0,'payload_bytes':sizes,'payload_sha256':hashes,'payload_files':{m:f'runs/dev_payloads/{seed}_{m}.pt' for m in sizes},'payload_replay_max_abs':replay,'compute_proxy_per_example':flop_proxies(),'resident_reconstructed_matrices_bytes':K*R*R*4,'angle_reconstruction_startup_givens':K*len(PAIRS),'latency':times,'errors':errs}

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=[60801,60802]);p.add_argument('--out',required=True);q=p.parse_args();torch.set_num_threads(1)
    d={'experiment_id':'MA-608','phase':'deterministic algebra/runtime screen','python':platform.python_version(),'torch':torch.__version__,'device':'cpu','worlds':{str(s):run(s) for s in q.seeds}}
    out=Path(q.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
if __name__=='__main__':main()
