from __future__ import annotations
import argparse,csv,hashlib,io,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
K=4;Z=8;NTRAIN=256;NHELD=64;WIDTH=64;UPDATES=3000

class PeriodicView(nn.Module):
    def __init__(self):
        super().__init__();self.map=nn.Linear(Z,3*K)
    def forward(self,x,z):
        raw=self.map(z).reshape(-1,3,K);amp=.5+.25*raw[:,0].tanh();freq=torch.arange(1,K+1,device=x.device,dtype=x.dtype)[None,:]+.2*raw[:,1].tanh();phase=.5*raw[:,2].tanh()
        return (amp*torch.sin(2*torch.pi*freq*x[:,None]+phase)).mean(-1)

class NativeHarmonic(PeriodicView):
    """Native direct amplitude/frequency/phase conditioning; algebraic alias control."""

class ConcatMLP(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(Z+1,WIDTH),nn.Tanh(),nn.Linear(WIDTH,WIDTH),nn.Tanh(),nn.Linear(WIDTH,1))
    def forward(self,x,z):return self.net(torch.cat((x[:,None],z),dim=-1)).squeeze(-1)

def make_world(seed):
    g=torch.Generator().manual_seed(seed+101);codes=torch.randn(NTRAIN+NHELD,Z,generator=g)
    mats=[torch.randn(Z,K,generator=g)/Z**.5 for _ in range(3)]
    amp=.5+.25*(codes@mats[0]).tanh();freq=torch.arange(1,K+1)[None,:]+.2*(codes@mats[1]).tanh();phase=.5*(codes@mats[2]).tanh()
    perm=torch.randperm(len(codes),generator=torch.Generator().manual_seed(seed+202));train_ix=perm[:NTRAIN];held_ix=perm[NTRAIN:]
    return codes,amp,freq,phase,train_ix,held_ix

def sample_x(seed,nfunc,npoint):
    g=torch.Generator().manual_seed(seed);return torch.rand(nfunc,npoint,generator=g)*2-1

def harmonic(x,amp,freq,phase):return (amp[:,None,:]*torch.sin(2*torch.pi*freq[:,None,:]*x[:,:,None]+phase[:,None,:])).mean(-1)
def target(x,codes,amp,freq,phase,ix):return harmonic(x,amp[ix],freq[ix],phase[ix])

def fit_shared(model,x,y,z,seed,lr,updates=UPDATES):
    torch.manual_seed(seed);opt=torch.optim.Adam(model.parameters(),lr=lr);g=torch.Generator().manual_seed(seed+303);start=time.perf_counter();nf,npnt=x.shape
    for _ in range(updates):
        fi=torch.randint(nf,(1024,),generator=g);pi=torch.randint(npnt,(1024,),generator=g);pred=model(x[fi,pi],z[fi]);loss=((pred-y[fi,pi])**2).mean();opt.zero_grad();loss.backward();opt.step()
    return time.perf_counter()-start

def fit_private(x,y,seed,updates=400):
    p=nn.Parameter(torch.zeros(x.shape[0],3*K));opt=torch.optim.Adam([p],lr=.03);g=torch.Generator().manual_seed(seed+404);start=time.perf_counter();nf,npnt=x.shape
    p.data[:,K:2*K]=torch.arange(1,K+1)[None,:]
    for _ in range(updates):
        fi=torch.randint(nf,(1024,),generator=g);pi=torch.randint(npnt,(1024,),generator=g);v=p[fi].reshape(-1,3,K);pred=(v[:,None,0,:]*torch.sin(2*torch.pi*v[:,None,1,:]*x[fi,pi,None]+v[:,None,2,:])).mean(-1);loss=((pred-y[fi,pi])**2).mean();opt.zero_grad();loss.backward();opt.step()
    return p.detach(),time.perf_counter()-start

def pack_model(model,codes,kind):
    arr={f'w_{k.replace(".","__")}':v.detach().cpu().numpy().astype(np.float16) for k,v in model.state_dict().items()};arr['codes']=codes.detach().cpu().numpy().astype(np.float16);arr['metadata_utf8']=np.frombuffer(json.dumps({'kind':kind,'latent_dim':Z,'components':K,'dtype':'float16'},sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8);b=io.BytesIO();np.savez_compressed(b,**arr);return b.getvalue()
def pack_values(values,kind):
    arr={'values':values.detach().cpu().numpy().astype(np.float16),'metadata_utf8':np.frombuffer(json.dumps({'kind':kind,'components':K,'dtype':'float16'},sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8)};b=io.BytesIO();np.savez_compressed(b,**arr);return b.getvalue()
def load(raw):
    a=np.load(io.BytesIO(raw),allow_pickle=False);kind=json.loads(bytes(a['metadata_utf8'].tolist()).decode())['kind'];v={'kind':kind}
    if 'codes' in a.files:v['codes']=torch.tensor(a['codes'].astype(np.float32))
    if 'values' in a.files:v['values']=torch.tensor(a['values'].astype(np.float32))
    if any(k.startswith('w_') for k in a.files):
        m=ConcatMLP() if kind=='concat' else PeriodicView();state={k[2:].replace('__','.'):torch.tensor(a[k].astype(np.float32)) for k in a.files if k.startswith('w_')};m.load_state_dict(state);v['model']=m
    return v
def infer(raw,x,kind):
    v=load(raw)
    with torch.no_grad():
        if kind in ('mirror','native','concat'):return v['model'](x.reshape(-1),v['codes'][:,None,:].expand(-1,x.shape[1],-1).reshape(-1,Z)).reshape(x.shape)
        c=v['values'].reshape(-1,3,K);return (c[:,None,0,:]*torch.sin(2*torch.pi*c[:,None,1,:]*x[:,:,None]+c[:,None,2,:])).mean(-1)
def nrmse(pred,y):
    den=((y-y.mean(1,keepdim=True))**2).mean(1).sqrt().clamp_min(1e-8);return float(((pred-y).pow(2).mean(1).sqrt()/den).mean())
def throughput(raw,x,kind):
    v=load(raw);xx=x[:,:256];
    if kind in ('mirror','native','concat'):
        z=v['codes'][:,None,:].expand(-1,xx.shape[1],-1).reshape(-1,Z);xf=xx.reshape(-1)
        def f():
            with torch.no_grad():return v['model'](xf,z)
    else:
        c=v['values'].reshape(-1,3,K)
        def f():
            with torch.no_grad():return (c[:,None,0,:]*torch.sin(2*torch.pi*c[:,None,1,:]*xx[:,:,None]+c[:,None,2,:])).mean(-1)
    for _ in range(4):f()
    st=time.perf_counter()
    for _ in range(20):f()
    return 20*xx.numel()/(time.perf_counter()-st)

def run(out,seeds=(41901,41902),train_n=128,query_n=512):
    out.mkdir(parents=True,exist_ok=True);rows=[];screens=[];torch.set_num_threads(1)
    for seed in seeds:
        codes,amp,freq,phase,tr,he=make_world(seed);allx=sample_x(seed+1,len(codes),train_n);ally=harmonic(allx,amp,freq,phase);xtr=allx[tr];ytr=ally[tr];ztr=codes[tr];xh=sample_x(seed+2,len(he),96);yh=target(xh,codes,amp,freq,phase,he);zh=codes[he];xq=sample_x(seed+3,len(he),query_n);yq=target(xq,codes,amp,freq,phase,he)
        torch.manual_seed(seed+505);mirror=PeriodicView();mt=fit_shared(mirror,xtr,ytr,ztr,seed+6,.01)
        torch.manual_seed(seed+505);native=NativeHarmonic();nt=fit_shared(native,xtr,ytr,ztr,seed+6,.01)
        torch.manual_seed(seed+506);concat=ConcatMLP();ct=fit_shared(concat,xtr,ytr,ztr,seed+7,.001)
        private,pt=fit_private(xh,yh,seed+8)
        oracle=torch.cat((amp[he],freq[he],phase[he]),dim=1)
        models={'mirror':(mirror,mt),'native':(native,nt),'concat':(concat,ct)};payloads={name:pack_model(m,zh,name) for name,(m,_) in models.items()};payloads['private']=pack_values(private,'private_harmonic');payloads['oracle']=pack_values(oracle,'oracle_harmonic')
        for name,raw in payloads.items():(out/f'dev{seed}_{name}.npz').write_bytes(raw)
        scores={name:nrmse(infer(raw,xq,name if name in ('mirror','native','concat') else 'table'),yq) for name,raw in payloads.items()};rates={name:throughput(raw,xq,name if name in ('mirror','native','concat') else 'table') for name,raw in payloads.items()}
        for name,raw in payloads.items():
            model_time=models[name][1] if name in models else pt if name=='private' else 0.;updates=UPDATES if name in models else 400 if name=='private' else 0;examples=NTRAIN*train_n if name in models else NHELD*96 if name=='private' else 0;digest=hashlib.sha256(raw).hexdigest();rows.append({'condition':'heldout_signal_zero_shot','world_or_seed':seed,'method':name,'serialized_bytes':len(raw),'train_tokens_or_examples':examples,'optimizer_updates':updates,'active_compute_proxy':(Z*3*K if name in ('mirror','native') else (Z+1)*WIDTH+WIDTH*WIDTH+WIDTH if name=='concat' else 3*K),'wall_time_s':round(model_time,6),'primary_metric':'heldout_NRMSE','primary_value':scores[name],'secondary_metric':'logical_queries_per_second','secondary_value':rates[name],'status_note':digest})
        alias=float((infer(payloads['mirror'],xq,'mirror')-infer(payloads['native'],xq,'native')).abs().max())<=1e-7
        screens.append({'seed':seed,'mirror_nrmse':scores['mirror'],'concat_nrmse':scores['concat'],'native_nrmse':scores['native'],'private_nrmse':scores['private'],'oracle_nrmse':scores['oracle'],'mirror_bytes':len(payloads['mirror']),'concat_bytes':len(payloads['concat']),'mirror_qps':rates['mirror'],'concat_qps':rates['concat'],'native_alias_exact':alias,'pass':scores['mirror']<=.05 and scores['mirror']<=.8*scores['concat'] and len(payloads['mirror'])<=.8*len(payloads['concat']) and rates['mirror']>=.8*rates['concat'] and not alias})
    with (out/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    result={'experiment_id':'MA-419','protocol_frozen':True,'fresh_accessed':False,'development_gate_passed':all(r['pass'] for r in screens),'seed_results':screens};(out/'screen.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'runs');a=p.parse_args();print(json.dumps(run(a.out),indent=2))
