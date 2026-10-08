from __future__ import annotations
import argparse,csv,hashlib,io,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
D=8;H=32;M=16;N0=32;EULER=64;RK4=256

class Field(nn.Module):
    def __init__(self):
        super().__init__();self.l1=nn.Linear(D,H);self.l2=nn.Linear(H,D)
    def forward(self,h):return self.l2(torch.tanh(self.l1(h)))

def make_world(seed):
    torch.manual_seed(seed);f=Field()
    with torch.no_grad():
        f.l1.weight.mul_(.10);f.l1.bias.mul_(.02);f.l2.weight.mul_(.035);f.l2.bias.mul_(.01)
    theta=torch.linspace(-.75,.75,M);g=torch.Generator().manual_seed(seed+10);x0=torch.randn(N0,D,generator=g)*.5
    return f,theta,x0

def rotation(theta):
    c=theta.cos();s=theta.sin();q=torch.zeros(len(theta),D,D);q[:,range(D),range(D)]=1
    for i in range(0,D,2):q[:,i,i]=c;q[:,i,i+1]=-s;q[:,i+1,i]=s;q[:,i+1,i+1]=c
    return q

def transformed(f,theta):
    q=rotation(theta);w1=torch.einsum('hd,mdk->mhk',f.l1.weight,q);w2=torch.einsum('mdk,kh->mdh',q.transpose(1,2),f.l2.weight);b2=torch.einsum('mdk,k->md',q.transpose(1,2),f.l2.bias);return w1,w2,b2

def pack(f,theta,kind):
    st=f.state_dict()
    if kind in ('mirror','native'):
        arr={'l1w':st['l1.weight'].numpy().astype(np.float16),'l1b':st['l1.bias'].numpy().astype(np.float16),'l2w':st['l2.weight'].numpy().astype(np.float16),'l2b':st['l2.bias'].numpy().astype(np.float16),'theta':theta.numpy().astype(np.float16),'ids':np.arange(M,dtype=np.uint8)}
        tag='shared_base_plus_angle_code'
    elif kind=='independent':
        w1,w2,b2=transformed(f,theta);arr={'w1':w1.detach().numpy().astype(np.float16),'w2':w2.detach().numpy().astype(np.float16),'b1':st['l1.bias'].numpy().astype(np.float16),'b2':b2.detach().numpy().astype(np.float16),'ids':np.arange(M,dtype=np.uint8)};tag='independent_transformed_fields'
    else:
        arr={'l1w':st['l1.weight'].numpy().astype(np.float16),'l1b':st['l1.bias'].numpy().astype(np.float16),'l2w':st['l2.weight'].numpy().astype(np.float16),'l2b':st['l2.bias'].numpy().astype(np.float16),'ids':np.arange(M,dtype=np.uint8)};tag='one_shared_field'
    arr['metadata_utf8']=np.frombuffer(json.dumps({'kind':tag,'state_dimension':D,'hidden':H,'dtype':'float16'},sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8);b=io.BytesIO();np.savez_compressed(b,**arr);return b.getvalue()

def load(raw):
    a=np.load(io.BytesIO(raw),allow_pickle=False);meta=json.loads(bytes(a['metadata_utf8'].tolist()).decode());kind=meta['kind'];v={'kind':kind}
    if kind=='independent_transformed_fields':
        v.update(w1=torch.tensor(a['w1'].astype(np.float32)),w2=torch.tensor(a['w2'].astype(np.float32)),b1=torch.tensor(a['b1'].astype(np.float32)),b2=torch.tensor(a['b2'].astype(np.float32)))
    else:
        f=Field();f.load_state_dict({'l1.weight':torch.tensor(a['l1w'].astype(np.float32)),'l1.bias':torch.tensor(a['l1b'].astype(np.float32)),'l2.weight':torch.tensor(a['l2w'].astype(np.float32)),'l2.bias':torch.tensor(a['l2b'].astype(np.float32))});v['field']=f
        if 'theta' in a.files:
            v['theta']=torch.tensor(a['theta'].astype(np.float32));q=rotation(v['theta']);w1,w2,b2=transformed(f,v['theta']);v.update(q=q,w1=w1,w2=w2,b1=f.l1.bias[None,:].expand(M,-1),b2=b2)
    return v

def basef(f,h):return f(h)
def field_eval(v,h,kind):
    if kind=='mirror':
        q=v['q'];qh=torch.einsum('mij,mnj->mni',q,h);out=v['field'](qh.reshape(-1,D)).reshape(M,N0,D);return torch.einsum('mji,mnj->mni',q,out)
    if kind in ('native','independent'):
        a=torch.einsum('mhd,mnd->mnh',v['w1'],h)+v['b1'][:,None,:];return torch.einsum('mdh,mnh->mnd',v['w2'],a.tanh())+v['b2'][:,None,:]
    return v['field'](h.reshape(-1,D)).reshape(M,N0,D)

def simulate_loaded(v,x0,kind,steps=EULER,integrator='euler'):
    h=x0[None,:,:].expand(M,-1,-1).clone();states=[h.clone()];dt=1/steps
    if integrator=='euler':
        for _ in range(steps):h=h+dt*field_eval(v,h,kind);states.append(h.clone())
    else:
        sub=RK4;dt=1/sub;stride=sub//64
        for i in range(sub):
            k1=field_eval(v,h,kind);k2=field_eval(v,h+dt*k1/2,kind);k3=field_eval(v,h+dt*k2/2,kind);k4=field_eval(v,h+dt*k3,kind);h=h+dt*(k1+2*k2+2*k3+k4)/6
            if (i+1)%stride==0:states.append(h.clone())
    return torch.stack(states)
def simulate(raw,x0,kind,steps=EULER,integrator='euler'):
    return simulate_loaded(load(raw),x0,kind,steps,integrator)

def rel_rmse(pred,ref):return float((torch.mean((pred-ref)**2).sqrt()/torch.mean(ref**2).sqrt().clamp_min(1e-9)))
def diversity(pred):
    final=pred[-1];d=[]
    for i in range(M):
        for j in range(i+1,M):d.append((final[i]-final[j]).pow(2).mean().sqrt())
    return float(torch.stack(d).mean())
def qps(raw,x0,kind):
    v=load(raw)
    for _ in range(5):simulate_loaded(v,x0,kind)
    st=time.perf_counter()
    for _ in range(15):simulate_loaded(v,x0,kind)
    return (15*M*N0*EULER)/(time.perf_counter()-st)
def setup_seconds(raw):st=time.perf_counter();load(raw);return time.perf_counter()-st

def run(out,seeds=(42401,42402)):
    out.mkdir(parents=True,exist_ok=True);rows=[];screens=[];torch.set_num_threads(1)
    for seed in seeds:
        f,theta,x0=make_world(seed);payloads={k:pack(f,theta,k) for k in ('mirror','native','independent','shared')}
        for k,p in payloads.items():(out/f'dev{seed}_{k}.npz').write_bytes(p)
        ref=simulate(payloads['mirror'],x0,'mirror',steps=RK4,integrator='rk4');results={};rates={}
        for name,raw in payloads.items():results[name]=simulate(raw,x0,name);rates[name]=qps(raw,x0,name)
        scores={name:rel_rmse(pred,ref) for name,pred in results.items()};div=diversity(results['mirror']);alias=float((results['mirror']-results['native']).abs().max())<=1e-7 and len(payloads['mirror'])==len(payloads['native'])
        for name,raw in payloads.items():
            digest=hashlib.sha256(raw).hexdigest();proxy=(D*H+H*D)+(24 if name=='mirror' else 0);rows.append({'condition':'sixteen_mode_ode_trajectory','world_or_seed':seed,'method':name,'serialized_bytes':len(raw),'train_tokens_or_examples':0,'optimizer_updates':0,'active_compute_proxy':proxy,'wall_time_s':round(setup_seconds(raw),6),'primary_metric':'relative_trajectory_RMSE_vs_RK4','primary_value':scores[name],'secondary_metric':'vector_field_evaluations_per_second','secondary_value':rates[name],'status_note':digest+';64-Euler-NFE'})
        screens.append({'seed':seed,'mirror_error':scores['mirror'],'native_error':scores['native'],'independent_error':scores['independent'],'shared_error':scores['shared'],'mirror_bytes':len(payloads['mirror']),'native_bytes':len(payloads['native']),'independent_bytes':len(payloads['independent']),'mirror_qps':rates['mirror'],'native_qps':rates['native'],'independent_qps':rates['independent'],'mode_diversity_rms':div,'native_exact_output_and_payload_size_alias':alias,'pass':scores['mirror']<=1e-3 and scores['mirror']<=1.05*scores['independent'] and len(payloads['mirror'])<=.8*len(payloads['independent']) and rates['mirror']>=.8*rates['native'] and div>1e-5 and not alias})
    with (out/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    result={'experiment_id':'MA-424','protocol_frozen':True,'fresh_accessed':False,'development_gate_passed':all(x['pass'] for x in screens),'seed_results':screens};(out/'screen.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'runs');a=p.parse_args();print(json.dumps(run(a.out),indent=2))
