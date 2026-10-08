"""MA-724 WDBC screen. Fresh metrics are available only with --stage fresh."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.func import functional_call

ROOT=Path(__file__).resolve().parents[1]
SEED=724; D=32; LAMBDAS=[0.1,1.0,10.0,100.0]; MC=100

class MirrorMLP(nn.Module):
    def __init__(self):
        super().__init__(); self.fc1=nn.Linear(30,D);self.fc2=nn.Linear(D,D);self.m=nn.Parameter(torch.zeros(D//2));self.head=nn.Linear(D,1)
    def features(self,x): return torch.relu(self.fc2(torch.relu(self.fc1(x))))
    def view(self,h,m=None):
        a=self.m if m is None else m; z=h.reshape(-1,D//2,2);c=torch.cos(a);s=torch.sin(a)
        return torch.stack((z[...,0]*c-z[...,1]*s,z[...,0]*s+z[...,1]*c),-1).reshape(-1,D)
    def logits_from_features(self,h,m=None,head_w=None,head_b=None):
        z=self.view(h,m)
        if head_w is None:return self.head(z).squeeze(-1)
        return (z*head_w.reshape(1,-1)).sum(-1)+head_b.reshape(())
    def forward(self,x):return self.logits_from_features(self.features(x))


def dataset():
    data=load_breast_cancer();X=data.data.astype(np.float32);y=data.target.astype(np.float32)
    Xtr,Xrest,ytr,yrest=train_test_split(X,y,test_size=.4,stratify=y,random_state=SEED)
    Xdev,Xfresh,ydev,yfresh=train_test_split(Xrest,yrest,test_size=.5,stratify=yrest,random_state=SEED+1)
    scaler=StandardScaler().fit(Xtr);Xtr=scaler.transform(Xtr).astype(np.float32);Xdev=scaler.transform(Xdev).astype(np.float32);Xfresh=scaler.transform(Xfresh).astype(np.float32)
    digest=hashlib.sha256(np.ascontiguousarray(np.column_stack([X,y])).tobytes()).hexdigest()
    return tuple(torch.tensor(z,dtype=torch.float32) for z in (Xtr,ytr,Xdev,ydev,Xfresh,yfresh)),scaler,digest


def metrics(probs,y):
    p=np.clip(np.asarray(probs),1e-7,1-1e-7);y=np.asarray(y)
    nll=float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))));brier=float(np.mean((p-y)**2));acc=float(np.mean((p>=.5)==y))
    ece=0.0
    for lo in np.linspace(0,1,11)[:-1]:
        mask=(p>=lo)&(p<(lo+.1 if lo<.9 else 1.000001))
        if mask.any():ece+=mask.mean()*abs(float(p[mask].mean())-float(y[mask].mean()))
    return {'nll':nll,'brier':brier,'ece10':float(ece),'accuracy':acc}


def fit_model(Xtr,ytr,Xdev,ydev,seed=SEED,max_epochs=300):
    torch.manual_seed(seed);m=MirrorMLP();opt=torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=1e-4)
    best=float('inf');best_state=None;best_epoch=0;stale=0;t0=time.perf_counter()
    for ep in range(max_epochs):
        m.train();opt.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(m(Xtr),ytr);loss.backward();opt.step()
        m.eval()
        with torch.no_grad():vl=float(nn.functional.binary_cross_entropy_with_logits(m(Xdev),ydev))
        if vl<best-1e-6:
            best=vl;best_epoch=ep+1;best_state={k:v.detach().clone() for k,v in m.state_dict().items()};stale=0
        else:stale+=1
        if stale>=40:break
    m.load_state_dict(best_state);return m,{'best_epoch':best_epoch,'updates':ep+1,'train_examples':len(Xtr)*(ep+1),'wall_s':time.perf_counter()-t0,'dev_map_nll':best}


def fisher_and_subnet(model,X):
    params=list(model.named_parameters());diag={k:torch.zeros_like(v) for k,v in params};Hm=torch.zeros(D//2,D//2);Hs=torch.zeros(D+1,D+1)
    model.eval()
    for i in range(len(X)):
        x=X[i:i+1];model.zero_grad(set_to_none=True);logit=model(x).squeeze();p=torch.sigmoid(logit).detach();w=p*(1-p)
        grads=torch.autograd.grad(logit,[v for _,v in params],retain_graph=False)
        gm=None
        for (name,_),g in zip(params,grads):
            diag[name]+=w*g.detach().square()
            if name=='m':gm=g.detach()
        h=model.features(x);z=model.view(h);aug=torch.cat([z,torch.ones(1,1)],1).squeeze(0).detach();Hs+=w*(aug[:,None]*aug[None,:])
        Hm+=w*(gm[:,None]*gm[None,:])
    hessian={'mirror_full':Hm,'subnet_full':Hs}
    return diag,hessian


def covariances(model,diag,hess,lam):
    full={k:1.0/(v+lam) for k,v in diag.items()}
    subnet=torch.linalg.inv(hess['subnet_full']+lam*torch.eye(D+1))
    mcov=torch.linalg.inv(hess['mirror_full']+lam*torch.eye(D//2))
    mdiag=1.0/(torch.diag(hess['mirror_full'])+lam)
    return full,subnet,mcov,mdiag


def posterior_probs(model,X,kind,cov,n=MC,seed=1):
    torch.manual_seed(seed);model.eval();out=[]
    with torch.no_grad():
        if kind=='map':return torch.sigmoid(model(X)).numpy()
        if kind in ('mirror_full','mirror_diag','subnet_full'):
            h=model.features(X)
            if kind=='subnet_full':
                C=cov;mean=torch.cat([model.head.weight.detach().flatten(),model.head.bias.detach().reshape(1)]);chol=torch.linalg.cholesky(C+1e-8*torch.eye(C.shape[0]));
                for _ in range(n):
                    beta=mean+chol@torch.randn_like(mean);logit=model.logits_from_features(h,head_w=beta[:-1],head_b=beta[-1]);out.append(torch.sigmoid(logit))
            elif kind=='mirror_full':
                C=cov;mean=model.m.detach();chol=torch.linalg.cholesky(C+1e-8*torch.eye(C.shape[0]));
                for _ in range(n):out.append(torch.sigmoid(model.logits_from_features(h,m=mean+chol@torch.randn_like(mean))))
            else:
                var=cov;mean=model.m.detach()
                for _ in range(n):out.append(torch.sigmoid(model.logits_from_features(h,m=mean+torch.sqrt(var)*torch.randn_like(mean))))
        elif kind=='full_diag':
            base={k:v.detach() for k,v in model.state_dict().items()}
            for _ in range(n):
                state={}
                for name,p in model.named_parameters():state[name]=p.detach()+torch.sqrt(cov[name])*torch.randn_like(p)
                for name,b in model.named_buffers():state[name]=b
                out.append(torch.sigmoid(functional_call(model,state,(X,))))
        return torch.stack(out).mean(0).numpy()


def save_base(path,model,scaler):
    state={k:v.detach().clone() for k,v in model.state_dict().items() if k!='m'}
    obj={'state_dict':state,'scaler_mean':torch.tensor(scaler.mean_,dtype=torch.float32),'scaler_scale':torch.tensor(scaler.scale_,dtype=torch.float32),'metadata':{'feature_dim':30,'hidden_dim':D,'shared_parameters':'fc1,fc2,head','sklearn':sklearn.__version__}}
    torch.save(obj,path);return os.path.getsize(path),hashlib.sha256(open(path,'rb').read()).hexdigest()

def save_payload(path,model,kind,cov,scaler,extra):
    obj={'method':kind,'m_mean':model.m.detach().clone(),'metadata':{'seed':SEED,'feature_dim':30,'hidden_dim':D,'view':'pairwise Givens angles before shared output readout','sklearn':sklearn.__version__},**extra}
    if kind=='full_diag':obj['posterior_variance']=cov
    elif kind in ('subnet_full','mirror_full'):obj['posterior_covariance']=cov
    elif kind=='mirror_diag':obj['posterior_variance_m']=cov
    elif kind=='ensemble5':obj['member_state_dicts']=extra['member_state_dicts'];obj['members']=5
    torch.save(obj,path);return os.path.getsize(path),hashlib.sha256(open(path,'rb').read()).hexdigest()


def run_dev():
    (Xtr,ytr,Xdev,ydev,Xfresh,yfresh),scaler,digest=dataset()
    model,trainlog=fit_model(Xtr,ytr,Xdev,ydev)
    diag,hess=fisher_and_subnet(model,Xtr)
    methods={'full_diag':'full_diag','subnet_full':'subnet_full','mirror_full':'mirror_full','mirror_diag':'mirror_diag'}
    best={}; devgrid={}
    for method in methods:
        scores=[]
        for lam in LAMBDAS:
            full,sub,mcov,mdiag=covariances(model,diag,hess,lam)
            cov={'full_diag':full,'subnet_full':sub,'mirror_full':mcov,'mirror_diag':mdiag}[method]
            p=posterior_probs(model,Xdev,method,cov,MC,SEED+int(lam*10));met=metrics(p,ydev.numpy());scores.append({'lambda':lam,**met})
        chosen=min(scores,key=lambda x:(x['nll'],x['ece10']));best[method]=chosen['lambda'];devgrid[method]=scores
    # Factor the shared physical network once; posterior files contain only per-view state/curvature.
    basebytes,basesha=save_base(ROOT/'source'/'shared_base.pt',model,scaler)
    # Save MAP and chosen posterior payloads. Fresh remains unopened.
    mapbytes,mapsha=save_payload(ROOT/'source'/'posterior_map.pt',model,'map',None,scaler,{})
    artifact={}
    for method,lam in best.items():
        full,sub,mcov,mdiag=covariances(model,diag,hess,lam);cov={'full_diag':full,'subnet_full':sub,'mirror_full':mcov,'mirror_diag':mdiag}[method]
        size,sha=save_payload(ROOT/'source'/f'posterior_{method}.pt',model,method,cov,scaler,{'prior_precision':lam,'curvature_examples':len(Xtr)})
        p=posterior_probs(model,Xdev,method,cov,MC,SEED+99);artifact[method]={'posterior_state_bytes':size,'total_payload_bytes':basebytes+size,'sha256':sha,'lambda':lam,'dev_metrics':metrics(p,ydev.numpy())}
    # Independent 5-model upper control, selected only on dev.
    ensemble=[];t0=time.perf_counter()
    for i in range(5):
        em,log=fit_model(Xtr,ytr,Xdev,ydev,seed=SEED+100+i);ensemble.append(em.state_dict())
    esize,esha=save_payload(ROOT/'source'/'posterior_ensemble5.pt',model,'ensemble5',None,scaler,{'member_state_dicts':ensemble,'members':5})
    with torch.no_grad():ep=np.mean([torch.sigmoid(functional_call(model,s,(Xdev,))).numpy() for s in ensemble],axis=0)
    artifact['ensemble5']={'posterior_state_bytes':esize,'total_payload_bytes':esize,'sha256':esha,'dev_metrics':metrics(ep,ydev.numpy()),'train_wall_s':time.perf_counter()-t0}
    # Chosen deterministic config is the only input to fresh evaluation.
    best_control=min(['full_diag','subnet_full'],key=lambda k:artifact[k]['dev_metrics']['nll'])
    frozen={'best_laplace_control':best_control,'experiment_id':'MA-724','dataset_sha256':digest,'prior_precision_by_method':best,'mc_samples':MC,'train_seed':SEED,'dev_seed':SEED+1,'fresh_seed_metrics_hidden':True,'methods':['map',*methods.keys(),'ensemble5']}
    (ROOT/'source'/'frozen_config.json').write_text(json.dumps(frozen,indent=2)+'\n')
    result={'shared_base_bytes':basebytes,'shared_base_sha256':basesha,'dataset_sha256':digest,'sklearn_version':sklearn.__version__,'sizes':{'train':len(Xtr),'dev':len(Xdev),'fresh_locked':len(Xfresh)},'train':trainlog,'map_metrics':metrics(torch.sigmoid(model(Xdev)).detach().numpy(),ydev.numpy()),'lambda_grid':devgrid,'chosen':artifact,'map_posterior_state_bytes':mapbytes,'map_total_payload_bytes':basebytes+mapbytes,'map_payload_sha256':mapsha,'curvature_examples':len(Xtr),'training_wall_s':trainlog['wall_s']}
    (ROOT/'source'/'development_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


def run_fresh():
    (Xtr,ytr,Xdev,ydev,Xfresh,yfresh),scaler,digest=dataset();frozen=json.loads((ROOT/'source'/'frozen_config.json').read_text());assert frozen['dataset_sha256']==digest
    results=[];latencies=[]
    for kind in frozen['methods']:
        obj=torch.load(ROOT/'source'/f'posterior_{kind}.pt',map_location='cpu',weights_only=False);model=MirrorMLP()
        if kind=='ensemble5':
            model.load_state_dict(obj['member_state_dicts'][0]);model.eval()
        else:
            base=torch.load(ROOT/'source'/'shared_base.pt',map_location='cpu',weights_only=False);state=dict(base['state_dict']);state['m']=obj['m_mean'];model.load_state_dict(state);model.eval()
        t0=time.perf_counter()
        if kind=='map':p=posterior_probs(model,Xfresh,'map',None,1,SEED+500)
        elif kind=='ensemble5':p=np.mean([torch.sigmoid(functional_call(model,s,(Xfresh,))).detach().numpy() for s in obj['member_state_dicts']],axis=0)
        else:
            cov=obj.get('posterior_variance',obj.get('posterior_covariance',obj.get('posterior_variance_m')));p=posterior_probs(model,Xfresh,kind,cov,MC,SEED+501)
        elapsed=time.perf_counter()-t0;met=metrics(p,yfresh.numpy());results.append({'method':kind,'metrics':met,'predictive_wall_s':elapsed,'posterior_state_bytes':os.path.getsize(ROOT/'source'/f'posterior_{kind}.pt'),'actual_payload_bytes':(os.path.getsize(ROOT/'source'/'shared_base.pt')+os.path.getsize(ROOT/'source'/f'posterior_{kind}.pt') if kind!='ensemble5' else os.path.getsize(ROOT/'source'/f'posterior_{kind}.pt'))});
        # Batch-1 posterior predictive latency on up to 64 fresh examples.
        sample=Xfresh[:min(64,len(Xfresh))];times=[]
        if kind=='ensemble5':
            members=[]
            for sd in obj['member_state_dicts']:
                member=MirrorMLP();member.load_state_dict(sd);member.eval();members.append(member)
            with torch.no_grad():
                for i in range(len(sample)):
                    a=time.perf_counter();_=[torch.sigmoid(member(sample[i:i+1])) for member in members];times.append(time.perf_counter()-a)
        else:
            cov=obj.get('posterior_variance',obj.get('posterior_covariance',obj.get('posterior_variance_m')))
            for i in range(len(sample)):
                a=time.perf_counter();_=posterior_probs(model,sample[i:i+1],kind,cov,MC,SEED+700+i);times.append(time.perf_counter()-a)
        latencies.append({'method':kind,'batch1_p50_s':float(np.quantile(times,.5)),'batch1_p95_s':float(np.quantile(times,.95)),'batch1_p99_s':float(np.quantile(times,.99))})
    out={'dataset_sha256':digest,'fresh_examples':len(Xfresh),'methods':results,'latencies':latencies,'fresh_accessed_once':True}
    (ROOT/'source'/'fresh_result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['dev','fresh'],default='dev');args=ap.parse_args()
    torch.set_num_threads(1)
    if args.stage=='dev':run_dev()
    else:run_fresh()
