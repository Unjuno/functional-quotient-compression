from __future__ import annotations
import argparse,csv,hashlib,io,json,time
from pathlib import Path
import numpy as np
import torch
E=8;D=8;NSEQ=24;LENGTH=128

def make_world(seed):
    g=torch.Generator().manual_seed(seed);gx=torch.Generator().manual_seed(seed+1)
    a=torch.randn(D,D,generator=g);a=a*(.65/torch.linalg.matrix_norm(a,ord=2));b=torch.randn(D,generator=g)*.2;c=torch.randn(D,generator=g)*.3;wg=torch.tensor(.8);bg=torch.tensor(-.3)
    theta=(torch.rand(E,D//2,generator=g)-.5)*1.6
    x=torch.randn(E,NSEQ,LENGTH,generator=gx)
    for t in range(11,LENGTH,29):x[:,:,t]+=torch.where(torch.rand(E,NSEQ,generator=gx)>.5,1.7,-1.7)
    return {'A':a,'B':b,'C':c,'wg':wg,'bg':bg},theta,x

def rotations(theta):
    q=torch.zeros(E,D,D);idx=torch.arange(D);q[:,idx,idx]=1
    for p in range(D//2):
        i=2*p;c=theta[:,p].cos();s=theta[:,p].sin();q[:,i,i]=c;q[:,i,i+1]=-s;q[:,i+1,i]=s;q[:,i+1,i+1]=c
    return q

def transformed(base,theta):
    q=rotations(theta);a=torch.einsum('eij,jk,ekl->eil',q.transpose(1,2),base['A'],q);b=torch.einsum('eij,j->ei',q.transpose(1,2),base['B']);c=torch.einsum('eij,j->ei',q.transpose(1,2),base['C']);return a,b,c

def metadata(kind):return np.frombuffer(json.dumps({'kind':kind,'experts':E,'state_dim':D,'dtype':'float16'},sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8)
def pack(base,theta,kind):
    ids=np.repeat(np.arange(E,dtype=np.uint8),NSEQ)
    if kind in ('mirror','native'):
        arr={k:v.detach().cpu().numpy().astype(np.float16) for k,v in base.items()};arr['theta']=theta.numpy().astype(np.float16);arr['route_ids']=ids;tag='shared_ssm_plus_expert_view_code'
    elif kind=='independent':
        a,b,c=transformed(base,theta);arr={'A':a.detach().numpy().astype(np.float16),'B':b.detach().numpy().astype(np.float16),'C':c.detach().numpy().astype(np.float16),'wg':np.full(E,float(base['wg']),dtype=np.float16),'bg':np.full(E,float(base['bg']),dtype=np.float16),'route_ids':ids};tag='independent_full_ssm_experts'
    else:
        arr={k:v.detach().cpu().numpy().astype(np.float16) for k,v in base.items()};arr['route_ids']=ids;tag='single_shared_ssm'
    arr['metadata_utf8']=metadata(tag);f=io.BytesIO();np.savez_compressed(f,**arr);return f.getvalue()

def load(raw,method):
    a=np.load(io.BytesIO(raw),allow_pickle=False);v={k:torch.tensor(a[k].astype(np.float32)) for k in a.files if k not in ('metadata_utf8','route_ids')};v['route_ids']=torch.tensor(a['route_ids'].astype(np.int64))
    if method=='mirror':v['Q']=rotations(v['theta'])
    elif method=='native':v['A_e'],v['B_e'],v['C_e']=transformed(v,v['theta'])
    return v

@torch.no_grad()
def simulate_values(v,x,method):
    h=torch.zeros(E,NSEQ,D);ys=[]
    for t in range(LENGTH):
        xt=x[:,:,t];u=torch.sigmoid(v['wg']*xt+v['bg'])*xt
        if method=='mirror':
            q=v['Q'];hv=torch.einsum('eij,enj->eni',q,h);hn=torch.einsum('ij,enj->eni',v['A'],hv)+v['B'][None,None,:]*u[:,:,None];h=torch.einsum('eji,enj->eni',q,hn);y=torch.einsum('i,eni->en',v['C'],hn)
        elif method=='native':
            h=torch.einsum('eij,enj->eni',v['A_e'],h)+v['B_e'][:,None,:]*u[:,:,None];y=torch.einsum('ei,eni->en',v['C_e'],h)
        elif method=='independent':
            h=torch.einsum('eij,enj->eni',v['A'],h)+v['B'][:,None,:]*u[:,:,None];y=torch.einsum('ei,eni->en',v['C'],h)
        else:
            h=torch.einsum('ij,enj->eni',v['A'],h)+v['B'][None,None,:]*u[:,:,None];y=torch.einsum('i,eni->en',v['C'],h)
        ys.append(y)
    return torch.stack(ys,dim=-1)
def simulate(raw,x,method):return simulate_values(load(raw,method),x,method)
def nrmse(pred,ref):
    den=((ref-ref.mean(-1,keepdim=True))**2).mean(-1).sqrt().clamp_min(1e-8);return float(((pred-ref).pow(2).mean(-1).sqrt()/den).mean())
def diversity(y):
    vals=[]
    for i in range(E):
        for j in range(i+1,E):vals.append((y[i]-y[j]).pow(2).mean().sqrt())
    return float(torch.stack(vals).mean())
def throughput(raw,x,method):
    v=load(raw,method)
    for _ in range(5):simulate_values(v,x,method)
    st=time.perf_counter()
    for _ in range(10):simulate_values(v,x,method)
    return 10*x.numel()/(time.perf_counter()-st)
def setup_time(raw,method):st=time.perf_counter();load(raw,method);return time.perf_counter()-st

def run(out,seeds=(43601,43602)):
    out.mkdir(parents=True,exist_ok=True);rows=[];screens=[];torch.set_num_threads(1)
    for seed in seeds:
        base,theta,x=make_world(seed);ref=simulate_values({**base,'Q':rotations(theta),'theta':theta},x,'mirror');payloads={m:pack(base,theta,m) for m in ('mirror','native','independent','shared')}
        for m,raw in payloads.items():(out/f'dev{seed}_{m}.npz').write_bytes(raw)
        preds={m:simulate(raw,x,m) for m,raw in payloads.items()};scores={m:nrmse(preds[m],ref) for m in payloads};rates={m:throughput(raw,x,m) for m,raw in payloads.items()};div=diversity(preds['mirror']);alias=payloads['mirror']==payloads['native'] and torch.max(torch.abs(preds['mirror']-preds['native']))<=1e-6
        for m,raw in payloads.items():
            digest=hashlib.sha256(raw).hexdigest();proxy=D*D+2*D+(8 if m=='mirror' else 0);rows.append({'condition':'routed_eight_expert_ssm','world_or_seed':seed,'method':m,'serialized_bytes':len(raw),'train_tokens_or_examples':0,'optimizer_updates':0,'active_compute_proxy':proxy,'wall_time_s':round(setup_time(raw,m),6),'primary_metric':'serialized_output_NRMSE_to_FP32_teacher','primary_value':scores[m],'secondary_metric':'tokens_per_second','secondary_value':rates[m],'status_note':digest+';recurrent linear SSM'})
        screens.append({'seed':seed,'mirror_nrmse':scores['mirror'],'native_nrmse':scores['native'],'independent_nrmse':scores['independent'],'shared_nrmse':scores['shared'],'mirror_bytes':len(payloads['mirror']),'native_bytes':len(payloads['native']),'independent_bytes':len(payloads['independent']),'mirror_tokens_s':rates['mirror'],'native_tokens_s':rates['native'],'mode_diversity_rms':div,'native_exact_payload_and_output_alias':bool(alias),'pass':scores['mirror']<=.01 and len(payloads['mirror'])<=.8*len(payloads['independent']) and rates['mirror']>=.8*rates['native'] and div>=.05 and not alias})
    with (out/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    result={'experiment_id':'MA-436','protocol_frozen':True,'fresh_accessed':False,'development_gate_passed':all(s['pass'] for s in screens),'seed_results':screens};(out/'screen.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'runs');a=p.parse_args();print(json.dumps(run(a.out),indent=2))
