from __future__ import annotations
import argparse,csv,hashlib,io,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROLES=12;D=16;NSEQ=32;LENGTH=256

def make_world(seed):
    g=torch.Generator().manual_seed(seed);gen=torch.Generator().manual_seed(seed+1)
    A=torch.linspace(-1.8,-.5,D);B=torch.randn(D,generator=g)*.18;C=torch.randn(D,generator=g)*.25
    wd=torch.tensor(.7);bd=torch.tensor(-.4);wb=torch.randn(D,generator=g)*.15;bb=torch.zeros(D);wc=torch.randn(D,generator=g)*.15;bc=torch.zeros(D)
    codes=torch.linspace(-.6,.6,ROLES)
    x=torch.randn(ROLES,NSEQ,LENGTH,generator=gen)
    for t in range(13,LENGTH,37):x[:,:,t]+=torch.where(torch.rand(ROLES,NSEQ,generator=gen)>.5,1.5,-1.5)
    return {'A':A,'B':B,'C':C,'wd':wd,'bd':bd,'wb':wb,'bb':bb,'wc':wc,'bc':bc},codes,x

def tag(v):
    return np.frombuffer(json.dumps(v,sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8)
def pack(base,codes,kind):
    if kind in ('mirror','native'):
        arrays={k:v.detach().cpu().numpy().astype(np.float16) for k,v in base.items()};arrays['codes']=codes.numpy().astype(np.float16);arrays['ids']=np.arange(ROLES,dtype=np.uint8);metadata={'kind':'shared_selective_ssm_plus_role_code','state_dim':D,'dtype':'float16'}
    elif kind=='independent':
        arrays={k:v.detach().cpu().numpy().astype(np.float16) for k,v in base.items()};arrays['A']=np.stack([base['A']+c for c in codes]).astype(np.float16)
        for k in ('B','C','wd','bd','wb','bb','wc','bc'):arrays[k]=np.broadcast_to(arrays[k],(ROLES,*arrays[k].shape)).copy()
        arrays['ids']=np.arange(ROLES,dtype=np.uint8);metadata={'kind':'independent_full_selective_ssm','state_dim':D,'dtype':'float16'}
    else:
        arrays={k:v.detach().cpu().numpy().astype(np.float16) for k,v in base.items()};arrays['ids']=np.arange(ROLES,dtype=np.uint8);metadata={'kind':'shared_unconditioned_selective_ssm','state_dim':D,'dtype':'float16'}
    arrays['metadata_utf8']=tag(metadata);b=io.BytesIO();np.savez_compressed(b,**arrays);return b.getvalue()

def load(raw,method):
    a=np.load(io.BytesIO(raw),allow_pickle=False);v={k:torch.tensor(a[k].astype(np.float32)) for k in a.files if k not in ('metadata_utf8','ids')};v['ids']=torch.tensor(a['ids'].astype(np.int64));return v

@torch.no_grad()
def simulate_values(v,codes,x,method):
    M,N,T=x.shape;h=torch.zeros(M,N,D);outputs=[]
    if method=='independent':
        alog=v['A'];B=v['B'][:,None,:];C=v['C'][:,None,:];wd=v['wd'][:,None,None];bd=v['bd'][:,None,None];wb=v['wb'][:,None,:];bb=v['bb'][:,None,:];wc=v['wc'][:,None,:];bc=v['bc'][:,None,:]
    else:
        alog=v['A'][None,:]+(codes[:,None] if method in ('mirror','native') else 0);B=v['B'][None,None,:];C=v['C'][None,None,:];wd=v['wd'];bd=v['bd'];wb=v['wb'];bb=v['bb'];wc=v['wc'];bc=v['bc']
    for t in range(T):
        xt=x[:,:,t,None];delta=F.softplus(xt*wd+bd);bt=B*torch.sigmoid(xt*wb+bb);ct=C*torch.sigmoid(xt*wc+bc);h=torch.exp(-torch.exp(alog)[:,None,:]*delta)*h+bt*xt;outputs.append((ct*h).sum(-1))
    return torch.stack(outputs,dim=-1)

def simulate(raw,x,method):
    v=load(raw,method);return simulate_values(v,v.get('codes',torch.zeros(ROLES)),x,method)
def nrmse(pred,y):
    den=((y-y.mean(-1,keepdim=True))**2).mean(-1).sqrt().clamp_min(1e-8);return float(((pred-y).pow(2).mean(-1).sqrt()/den).mean())
def diversity(y):
    vals=[]
    for i in range(ROLES):
        for j in range(i+1,ROLES):vals.append((y[i]-y[j]).pow(2).mean().sqrt())
    return float(torch.stack(vals).mean())
def throughput(raw,x,method):
    v=load(raw,method);codes=v.get('codes',torch.zeros(ROLES));
    for _ in range(5):simulate_values(v,codes,x,method)
    st=time.perf_counter()
    for _ in range(10):simulate_values(v,codes,x,method)
    return 10*x.numel()/(time.perf_counter()-st)
def setup_time(raw,method):st=time.perf_counter();load(raw,method);return time.perf_counter()-st

def run(out,seeds=(43401,43402)):
    out.mkdir(parents=True,exist_ok=True);rows=[];screens=[];torch.set_num_threads(1)
    for seed in seeds:
        base,codes,x=make_world(seed);reference=simulate_values(base,codes,x,'mirror');payloads={m:pack(base,codes,m) for m in ('mirror','native','independent','shared')}
        for m,raw in payloads.items():(out/f'dev{seed}_{m}.npz').write_bytes(raw)
        preds={m:simulate(raw,x,m) for m,raw in payloads.items()};scores={m:nrmse(preds[m],reference) for m in payloads};rates={m:throughput(raw,x,m) for m,raw in payloads.items()};div=diversity(preds['mirror']);alias=payloads['mirror']==payloads['native'] and torch.max(torch.abs(preds['mirror']-preds['native']))==0
        for m,raw in payloads.items():
            digest=hashlib.sha256(raw).hexdigest();proxy=7*D+4;rows.append({'condition':'selective_ssm_role_sequences','world_or_seed':seed,'method':m,'serialized_bytes':len(raw),'train_tokens_or_examples':0,'optimizer_updates':0,'active_compute_proxy':proxy,'wall_time_s':round(setup_time(raw,m),6),'primary_metric':'serialized_output_NRMSE_to_FP32_teacher','primary_value':scores[m],'secondary_metric':'tokens_per_second','secondary_value':rates[m],'status_note':digest+';input-selective recurrence'})
        screens.append({'seed':seed,'mirror_nrmse':scores['mirror'],'native_nrmse':scores['native'],'independent_nrmse':scores['independent'],'shared_nrmse':scores['shared'],'mirror_bytes':len(payloads['mirror']),'native_bytes':len(payloads['native']),'independent_bytes':len(payloads['independent']),'mirror_tokens_s':rates['mirror'],'native_tokens_s':rates['native'],'mode_diversity_rms':div,'native_exact_output_and_payload_alias':bool(alias),'pass':scores['mirror']<=.01 and len(payloads['mirror'])<=.8*len(payloads['independent']) and rates['mirror']>=.8*rates['native'] and div>=.01 and not alias})
    with (out/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    result={'experiment_id':'MA-434','protocol_frozen':True,'fresh_accessed':False,'development_gate_passed':all(s['pass'] for s in screens),'seed_results':screens};(out/'screen.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'runs');a=p.parse_args();print(json.dumps(run(a.out),indent=2))
