#!/usr/bin/env python3
"""MA-612 signature x execution-code factorization."""
import argparse,hashlib,io,json,platform,time
from pathlib import Path
import torch
S,P,N=4,16,64
FREQ=torch.tensor([1.,2.,3.,4.]);PHASE=torch.arange(P)*2*torch.pi/P-torch.pi
PAIRS=[(s,p) for s in range(S) for p in range(P)]
HELD={i for i,(s,p) in enumerate(PAIRS) if (s+p)%4==0}

def pred(x,s,phi):return torch.sin(FREQ[s]*x+phi)
def payload(mode,seed):
    g=torch.Generator().manual_seed(seed);noise=torch.rand(N,generator=g)*0
    sig=torch.tensor([s for s,p in PAIRS],dtype=torch.uint8);phase=torch.tensor([p for s,p in PAIRS],dtype=torch.uint8)
    if mode=='unfactorized':state={'sig':sig,'codes':torch.stack((torch.cos(PHASE[phase.long()]),torch.sin(PHASE[phase.long()])),1).float()}
    elif mode=='factor_coeff':state={'freq':FREQ.clone(),'phase_codes':torch.stack((torch.cos(PHASE),torch.sin(PHASE)),1).float(),'pair_refs':torch.stack((sig,phase),1)}
    else:state={'freq':FREQ.clone(),'phase_angles':PHASE.float(),'pair_refs':torch.stack((sig,phase),1)}
    b=io.BytesIO();torch.save({'mode':mode,'state':state,'meta':{'seed':seed,'n_signatures':S,'n_views':P,'n_functions':N,'format':'MA612-v1'}},b);return b.getvalue()
def decode(blob,x):
    d=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False);m=d['mode'];st=d['state'];ys=[];route=[]
    for i in range(N):
        s,p=PAIRS[i];pi=p
        if m=='unfactorized':
            si=int(st['sig'][i]);c=st['codes'][i];y=c[0]*torch.sin(FREQ[si]*x)+c[1]*torch.cos(FREQ[si]*x)
        elif m=='factor_coeff':
            si,pi=map(int,st['pair_refs'][i]);c=st['phase_codes'][pi];y=c[0]*torch.sin(st['freq'][si]*x)+c[1]*torch.cos(st['freq'][si]*x)
        else:
            si,pi=map(int,st['pair_refs'][i]);y=torch.sin(st['freq'][si]*x+st['phase_angles'][pi])
        ys.append(y);route.append((si,pi))
    return torch.stack(ys),route
def run(seed):
    x=torch.linspace(-torch.pi,torch.pi,256);target=torch.stack([pred(x,s,PHASE[p]) for s,p in PAIRS]);out={};blobs={};root=Path(__file__).resolve().parents[1]/'runs/dev_payloads';root.mkdir(parents=True,exist_ok=True)
    for m in ('unfactorized','factor_coeff','mirror'):
        b=payload(m,seed);blobs[m]=b;path=root/f'{seed}_{m}.pt';path.write_bytes(b);y,route=decode(b,x);err=(y-target).abs().max().item();held=torch.tensor(sorted(HELD));heldmse=((y[held]-target[held])**2).mean().item()/target[held].var().item();out[m]={'payload_bytes':len(b),'payload_sha256':hashlib.sha256(b).hexdigest(),'payload_file':str(path.relative_to(Path(__file__).resolve().parents[1])),'max_abs_error':err,'heldout_nrmse2':heldmse,'routing_accuracy':sum(int(a==b) for a,b in zip(route,PAIRS))/N,'heldout_pairings':len(HELD),'examples':N*len(x),'optimizer_updates':0,'query_arithmetic_proxy_per_example':{'unfactorized':3,'factor_coeff':3,'mirror':2}[m],'query_trig_ops_per_example':{'unfactorized':2,'factor_coeff':2,'mirror':1}[m]}
    latency={}
    for m,b in blobs.items():
        decode(b,x);start=time.perf_counter()
        for _ in range(30):decode(b,x)
        latency[m]=(time.perf_counter()-start)*1000/30
    out['latency_ms_per_16384']=latency
    return out
def main():
    a=argparse.ArgumentParser();a.add_argument('--seeds',type=int,nargs='+',default=[61201,61202]);a.add_argument('--out',required=True);q=a.parse_args();torch.set_num_threads(1)
    d={'experiment_id':'MA-612','phase':'deterministic exhaustive pairing screen','python':platform.python_version(),'torch':torch.__version__,'device':'cpu','worlds':{str(s):run(s) for s in q.seeds}}
    p=Path(q.out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
if __name__=='__main__':main()
