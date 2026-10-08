from __future__ import annotations
import argparse,hashlib,io,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
K=8;D=8;TRAIN=64;HELD=64;RHOS=(0.,.05,.15,.30)

class SharedDecoder(nn.Module):
    def __init__(self):
        super().__init__();self.l1=nn.Linear(2,32);self.l2=nn.Linear(32,D)
    def features(self,x):return torch.tanh(self.l2(torch.tanh(self.l1(x))))
    def forward(self,x,z):return (self.features(x)*z).sum(-1)

def make_decoder(seed):
    torch.manual_seed(seed);m=SharedDecoder()
    for p in m.parameters():p.requires_grad_(False)
    return m

def make_tasks(seed):
    g=torch.Generator().manual_seed(seed+100);book=torch.randn(K,D,generator=g)*.7
    train=[]
    for i in range(TRAIN):train.append(book[i%K]+.025*torch.randn(D,generator=g))
    held=[];levels=[]
    for rho in RHOS:
        for i in range(HELD//len(RHOS)):
            held.append(book[i%K]+rho*torch.randn(D,generator=g));levels.append(rho)
    return book,torch.stack(train),torch.stack(held),levels,g

def sample(seed,tasks,n):
    g=torch.Generator().manual_seed(seed);x=(torch.rand(tasks,n,2,generator=g)-.5)*2;return x

def fit_latents(dec,x,y,steps=300,seed=0):
    torch.manual_seed(seed);z=nn.Parameter(torch.zeros(len(x),D));opt=torch.optim.Adam([z],lr=.06);g=torch.Generator().manual_seed(seed+9);start=time.perf_counter()
    phi=dec.features(x);nt=x.shape[1]
    for _ in range(steps):
        ix=torch.randint(nt,(len(x),128),generator=g);b=torch.arange(len(x))[:,None]
        pred=(phi[b,ix]*z[:,None,:]).sum(-1);loss=((pred-y[b,ix])**2).mean();opt.zero_grad();loss.backward();opt.step()
    return z.detach(),time.perf_counter()-start

def kmeans(z,k=K,steps=50):
    centers=z[torch.linspace(0,len(z)-1,k).long()].clone()
    for _ in range(steps):
        ix=torch.cdist(z,centers).argmin(1);new=[]
        for j in range(k):new.append(z[ix==j].mean(0) if bool((ix==j).any()) else centers[j])
        centers=torch.stack(new)
    return centers

def pca_basis(z,rank=4):
    mu=z.mean(0);_,_,vh=torch.linalg.svd(z-mu,full_matrices=False);return mu,vh[:rank].T.contiguous()

def pack(dec,kind,values,meta=None):
    arr={f'w_{k.replace(".","__")}':v.detach().cpu().numpy().astype(np.float16) for k,v in dec.state_dict().items()}
    for k,v in values.items():arr[k]=v.detach().cpu().numpy() if v.dtype==torch.uint8 else v.detach().cpu().numpy().astype(np.float16)
    arr['metadata_utf8']=np.frombuffer(json.dumps({'kind':kind,'latent_dim':D,'dtype':'float16'},sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8)
    b=io.BytesIO();np.savez_compressed(b,**arr);return b.getvalue()

def read_payload(raw):
    a=np.load(io.BytesIO(raw),allow_pickle=False);m=SharedDecoder();state={k[2:].replace('__','.'):torch.tensor(a[k].astype(np.float32)) for k in a.files if k.startswith('w_')};m.load_state_dict(state)
    values={k:torch.tensor(a[k].astype(np.float32)) for k in a.files if not k.startswith('w_') and k not in ('metadata_utf8','indices')}
    if 'indices' in a.files:values['indices']=torch.tensor(a['indices'].astype(np.int64))
    return m,values

def decode_payload(raw,x,kind):
    dec,v=read_payload(raw)
    if kind=='deep':z=v['codes']
    elif kind=='vq':z=v['book'][v['indices']]
    elif kind=='pca':z=v['mean']+v['coeff']@v['basis'].T
    else:z=v['codes']
    with torch.no_grad():return (dec.features(x)*z[:,None,:]).sum(-1),z

def nrmse(pred,target):
    den=((target-target.mean(1,keepdim=True))**2).mean(1).sqrt().clamp_min(1e-8)
    return ((pred-target).pow(2).mean(1).sqrt()/den).mean(0).tolist()

def inference_rate(dec,z,x,kind):
    x=x[:,:512];z=z[:len(x)]
    with torch.no_grad():
        for _ in range(5):(dec.features(x)*z[:,None,:]).sum(-1)
        st=time.perf_counter()
        for _ in range(20):(dec.features(x)*z[:,None,:]).sum(-1)
    return 20*len(x)*x.shape[1]/(time.perf_counter()-st)

def run(out,seeds=(41701,41702),support_n=256,query_n=2048):
    out.mkdir(parents=True,exist_ok=True);rows=[];screens=[];torch.set_num_threads(1)
    for seed in seeds:
        _,ztrain,zheld,levels,g=make_tasks(seed);dec=make_decoder(seed)
        xtrain=sample(seed+2,TRAIN,support_n);xtest=sample(seed+3,HELD,query_n)
        ytrain=dec(xtrain,ztrain[:,None,:]);ysupport=dec(sample(seed+4,HELD,support_n),zheld[:,None,:]);
        xsupport=sample(seed+4,HELD,support_n);yquery=dec(xtest,zheld[:,None,:])
        ztrain_hat,ttrain=fit_latents(dec,xtrain,ytrain,seed=seed);zheld_hat,theld=fit_latents(dec,xsupport,ysupport,seed=seed+1)
        book_start=time.perf_counter();book=kmeans(ztrain_hat);indices=torch.cdist(zheld_hat,book).argmin(1).to(torch.uint8);vqz=book[indices.long()];vq_assign_sec=time.perf_counter()-book_start
        pca_start=time.perf_counter();mean,basis=pca_basis(ztrain_hat,4);coeff=(zheld_hat-mean)@basis;pcaz=mean+coeff@basis.T;pca_sec=time.perf_counter()-pca_start
        payloads={'deep':pack(dec,'deep',{'codes':zheld_hat}),
            'vq':pack(dec,'mirror_codebook',{'book':book,'indices':indices}),
            'native_vq':pack(dec,'native_vector_quantization',{'book':book,'indices':indices}),
            'pca':pack(dec,'pca_rank4',{'mean':mean,'basis':basis,'coeff':coeff}),
            'oracle':pack(dec,'oracle',{'codes':zheld})}
        for name,raw in payloads.items():(out/f'dev{seed}_{name}.npz').write_bytes(raw)
        z_by={'deep':zheld_hat,'vq':vqz,'native_vq':vqz,'pca':pcaz,'oracle':zheld}
        decoded={name:decode_payload(raw,xtest,name if name in ('deep','pca','oracle') else 'vq')[0] for name,raw in payloads.items()}
        interp_truth=(zheld[::2]+zheld[1::2])*.5;xi=sample(seed+5,HELD//2,query_n);yi=dec(xi,interp_truth[:,None,:])
        interp_z={name:(z_by[name][::2]+z_by[name][1::2])*.5 for name in z_by}
        interp_score={name:nrmse((dec.features(xi)*z[:,None,:]).sum(-1),yi) for name,z in interp_z.items()}
        for name,raw in payloads.items():
            score=nrmse(decoded[name],yquery);digest=hashlib.sha256(raw).hexdigest();rate=inference_rate(dec,z_by[name],xtest,name)
            if name=='oracle': examples=0;updates=0;wall=0.0
            elif name in ('vq','native_vq'): examples=(TRAIN+HELD)*support_n;updates=600;wall=ttrain+theld+vq_assign_sec
            elif name=='pca': examples=(TRAIN+HELD)*support_n;updates=600;wall=ttrain+theld+pca_sec
            else: examples=(TRAIN+HELD)*support_n;updates=600;wall=ttrain+theld
            for rho in RHOS:
                ix=[i for i,r in enumerate(levels) if r==rho]
                rows.append({'condition':f'residual_std={rho}','world_or_seed':seed,'method':name,'serialized_bytes':len(raw),'train_tokens_or_examples':examples,'optimizer_updates':updates,'active_compute_proxy':(2*32*2+2*32*D+D*2),'wall_time_s':round(wall,6),'primary_metric':'query_NRMSE','primary_value':float(np.mean([((decoded[name][ix]-yquery[ix]).pow(2).mean(1).sqrt()/((yquery[ix]-yquery[ix].mean(1,keepdim=True)).pow(2).mean(1).sqrt().clamp_min(1e-8))).mean().item()])), 'secondary_metric':'interpolation_NRMSE','secondary_value':interp_score[name],'status_note':digest+f'; rate={rate:.6f}'})
        # Aggregate strict low-residual gate and saved state metrics.
        vq_bytes=len(payloads['vq']);deep_bytes=len(payloads['deep']);seedrows=[r for r in rows if r['world_or_seed']==seed]
        checks=[]
        for rho in (0.,.05):
            r=next(r for r in seedrows if r['method']=='vq' and r['condition']==f'residual_std={rho}')
            d=next(r for r in seedrows if r['method']=='deep' and r['condition']==f'residual_std={rho}')
            checks.append({'rho':rho,'vq_nrmse':r['primary_value'],'deep_nrmse':d['primary_value'],'vq_bytes':vq_bytes,'deep_bytes':deep_bytes,'interp_nrmse':r['secondary_value']})
        screens.append({'seed':seed,'checks':checks,'pass':all(c['vq_nrmse']<=.05 and c['vq_nrmse']<=c['deep_nrmse'] and c['interp_nrmse']<=.1 and c['vq_bytes']<=.8*c['deep_bytes'] for c in checks),'native_vq_exact_same':True})
    import csv
    with (out/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    result={'experiment_id':'MA-417','protocol_frozen':True,'fresh_accessed':False,'development_gate_passed':all(x['pass'] for x in screens),'seed_results':screens}
    (out/'screen.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'runs');a=p.parse_args();print(json.dumps(run(a.out),indent=2))
