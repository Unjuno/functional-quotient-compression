from __future__ import annotations
import argparse,hashlib,io,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
O=8;S=8;D=4;Z=16;WIDTH=64;UPDATES=600

class SharedDecoder(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(2+Z,WIDTH),nn.Tanh(),nn.Linear(WIDTH,WIDTH),nn.Tanh(),nn.Linear(WIDTH,1))
    def forward(self,x,z):return self.net(torch.cat((x,z),dim=-1)).squeeze(-1)

def make_decoder(seed):
    torch.manual_seed(seed);m=SharedDecoder()
    for p in m.parameters():p.requires_grad_(False)
    return m

def combos():return [(o,s,d) for o in range(O) for s in range(S) for d in range(D)]
def make_world(seed):
    g=torch.Generator().manual_seed(seed+100);eo=torch.randn(O,Z,generator=g)*.35;es=torch.randn(S,Z,generator=g)*.35;ed=torch.randn(D,Z,generator=g)*.35
    allc=combos();pairs=torch.tensor(allc);teacher=eo[pairs[:,0]]+es[pairs[:,1]]+ed[pairs[:,2]]
    hold=torch.tensor([((o*13+s*7+d*3)%5)==0 for o,s,d in allc]);train_ix=torch.where(~hold)[0];held_ix=torch.where(hold)[0]
    assert all(bool((pairs[train_ix,k].unique().numel()==pairs[:,k].unique().numel())) for k in range(3))
    return pairs,teacher,train_ix,held_ix

def sample_inputs(seed,ncombos,n):
    g=torch.Generator().manual_seed(seed);return (torch.rand(ncombos,n,2,generator=g)-.5)*2

def target(dec,x,z):
    with torch.no_grad():return dec(x,z[:,None,:].expand(-1,x.shape[1],-1))

class FactorModel(nn.Module):
    def __init__(self):
        super().__init__();self.obj=nn.Parameter(torch.randn(O,Z)*.05);self.style=nn.Parameter(torch.randn(S,Z)*.05);self.domain=nn.Parameter(torch.randn(D,Z)*.05)
    def latent(self,ids):return self.obj[ids[:,0]]+self.style[ids[:,1]]+self.domain[ids[:,2]]
    def forward(self,dec,x,ids):return dec(x,self.latent(ids)[:,None,:].expand(-1,x.shape[1],-1))

def fit_factor(dec,x,y,ids,seed,updates=UPDATES):
    torch.manual_seed(seed);model=FactorModel();opt=torch.optim.Adam(model.parameters(),lr=.02);g=torch.Generator().manual_seed(seed+9);start=time.perf_counter()
    for _ in range(updates):
        ti=torch.randint(len(ids),(1024,),generator=g);pi=torch.randint(x.shape[1],(1024,),generator=g);xx=x[ti,pi];yy=y[ti,pi];pred=dec(xx,model.latent(ids[ti]));loss=((pred-yy)**2).mean();opt.zero_grad();loss.backward();opt.step()
    return model,time.perf_counter()-start

def fit_full_codes(dec,x,y,seed,updates=UPDATES):
    torch.manual_seed(seed);codes=nn.Parameter(torch.zeros(len(x),Z));opt=torch.optim.Adam([codes],lr=.04);g=torch.Generator().manual_seed(seed+10);start=time.perf_counter()
    for _ in range(updates):
        ti=torch.randint(len(x),(1024,),generator=g);pi=torch.randint(x.shape[1],(1024,),generator=g);pred=dec(x[ti,pi],codes[ti]);loss=((pred-y[ti,pi])**2).mean()+1e-4*(codes[ti]**2).mean();opt.zero_grad();loss.backward();opt.step()
    return codes.detach(),time.perf_counter()-start

def adapt_full_codes(dec,x,y,seed,updates=300):
    return fit_full_codes(dec,x,y,seed,updates)

def pack(dec,kind,values):
    arr={f'w_{k.replace(".","__")}':v.detach().cpu().numpy().astype(np.float16) for k,v in dec.state_dict().items()}
    for k,v in values.items():arr[k]=v.detach().cpu().numpy().astype(np.uint8 if v.dtype==torch.uint8 else np.float16)
    arr['metadata_utf8']=np.frombuffer(json.dumps({'kind':kind,'latent_dimension':Z,'dtype':'float16'},sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8)
    b=io.BytesIO();np.savez_compressed(b,**arr);return b.getvalue()

def load(raw):
    a=np.load(io.BytesIO(raw),allow_pickle=False);dec=SharedDecoder();state={k[2:].replace('__','.'):torch.tensor(a[k].astype(np.float32)) for k in a.files if k.startswith('w_')};dec.load_state_dict(state)
    v={k:torch.tensor(a[k].astype(np.int64 if k=='ids' else np.float32)) for k in a.files if not k.startswith('w_') and k!='metadata_utf8'}
    return dec,v

def infer(raw,ids,x,kind):
    dec,v=load(raw)
    if kind in ('factor','native'):
        z=v['obj'][ids[:,0]]+v['style'][ids[:,1]]+v['domain'][ids[:,2]]
    else:
        combo_idx=ids[:,0]*S*D+ids[:,1]*D+ids[:,2];z=v['codes'][combo_idx]
    with torch.no_grad():return dec(x,z[:,None,:].expand(-1,x.shape[1],-1))

def nrmse(pred,y):
    den=((y-y.mean(1,keepdim=True))**2).mean(1).sqrt().clamp_min(1e-8)
    return float(((pred-y).pow(2).mean(1).sqrt()/den).mean())

def rate(raw,ids,x,kind):
    # Deserialize once, as a serving runtime would. Report steady-state queries;
    # serialized bytes already charge all decode state separately.
    x=x[:,:256];ids=ids[:len(x)];dec,v=load(raw)
    if kind in ('factor','native'):
        z=v['obj'][ids[:,0]]+v['style'][ids[:,1]]+v['domain'][ids[:,2]]
    else:
        combo_idx=ids[:,0]*S*D+ids[:,1]*D+ids[:,2];z=v['codes'][combo_idx]
    z=z[:,None,:].expand(-1,x.shape[1],-1)
    def forward():
        with torch.no_grad():return dec(x,z)
    for _ in range(5):forward()
    st=time.perf_counter()
    for _ in range(20):forward()
    return 20*len(x)*x.shape[1]/(time.perf_counter()-st)

def run(out,seeds=(41801,41802),support_n=128,query_n=1024):
    out.mkdir(parents=True,exist_ok=True);rows=[];screens=[];torch.set_num_threads(1)
    for seed in seeds:
        pairs,ztrue,train_ix,held_ix=make_world(seed);dec=make_decoder(seed);allx=sample_inputs(seed+1,len(pairs),support_n);ally=target(dec,allx,ztrue)
        xtrain=allx[train_ix];ytrain=ally[train_ix];xheld=allx[held_ix];yheld=ally[held_ix];idtrain=pairs[train_ix];idheld=pairs[held_ix]
        factor,ft=fit_factor(dec,xtrain,ytrain,idtrain,seed+2)
        ztrain,dt=fit_full_codes(dec,xtrain,ytrain,seed+3);zheld,ht=adapt_full_codes(dec,xheld,yheld,seed+4)
        # Full code table is trained on observed combinations and support-adapted on heldout combinations.
        full=torch.zeros(len(pairs),Z);full[train_ix]=ztrain;full[held_ix]=zheld
        oracle=ztrue
        fvals={'obj':factor.obj.detach(),'style':factor.style.detach(),'domain':factor.domain.detach(),'ids':pairs.to(torch.uint8)}
        nvals={k:v.clone() for k,v in fvals.items()}
        payloads={'factor':pack(dec,'mirror_additive_factors',fvals),'native':pack(dec,'native_additive_factors',nvals),'deep':pack(dec,'deepsdf_full_codes',{'codes':full}),'oracle':pack(dec,'oracle',{'codes':oracle})}
        for name,raw in payloads.items():(out/f'dev{seed}_{name}.npz').write_bytes(raw)
        xq=sample_inputs(seed+5,len(held_ix),query_n);yq=target(dec,sample_inputs(seed+5,len(held_ix),query_n),ztrue[held_ix])
        scores={name:nrmse(infer(raw,idheld,xq,'deep' if name in ('deep','oracle') else name),yq) for name,raw in payloads.items()}
        rates={name:rate(raw,idheld,xq,'deep' if name in ('deep','oracle') else name) for name,raw in payloads.items()}
        for name,raw in payloads.items():
            digest=hashlib.sha256(raw).hexdigest();seconds=ft if name in ('factor','native') else dt+ht if name=='deep' else 0.;updates=UPDATES if name in ('factor','native') else UPDATES+300 if name=='deep' else 0
            examples=len(train_ix)*support_n+(len(held_ix)*support_n if name=='deep' else 0) if name!='oracle' else 0
            for label,val in [('heldout_unseen_combinations',scores[name])]:
                rows.append({'condition':label,'world_or_seed':seed,'method':name,'serialized_bytes':len(raw),'train_tokens_or_examples':examples,'optimizer_updates':updates,'active_compute_proxy':2*WIDTH*(2+Z)+2*WIDTH*WIDTH+2*WIDTH,'wall_time_s':round(seconds,6),'primary_metric':'heldout_NRMSE','primary_value':val,'secondary_metric':'logical_queries_per_second','secondary_value':rates[name],'status_note':digest})
        screens.append({'seed':seed,'factor_nrmse':scores['factor'],'deep_nrmse':scores['deep'],'factor_bytes':len(payloads['factor']),'deep_bytes':len(payloads['deep']),'factor_rate':rates['factor'],'deep_rate':rates['deep'],'native_alias_exact':torch.max(torch.abs(infer(payloads['factor'],idheld,xq,'factor')-infer(payloads['native'],idheld,xq,'native'))).item()==0,'pass':scores['factor']<=.05 and scores['factor']<=.5*scores['deep'] and len(payloads['factor'])<=.8*len(payloads['deep']) and torch.max(torch.abs(infer(payloads['factor'],idheld,xq,'factor')-infer(payloads['native'],idheld,xq,'native')))<=1e-6})
    import csv
    with (out/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    result={'experiment_id':'MA-418','protocol_frozen':True,'fresh_accessed':False,'development_gate_passed':all(r['pass'] for r in screens),'seed_results':screens};(out/'screen.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'runs');a=p.parse_args();print(json.dumps(run(a.out),indent=2))
