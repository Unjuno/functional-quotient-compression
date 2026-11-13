"""Locked synthetic threshold-specialist SNN screen for MA-968."""
import argparse,csv,hashlib,io,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
Q,D=8,16
NTRAIN,NDEV,NTEST=4096,1024,2048
DEV_SEEDS=(96801,)
FRESH_SEEDS=(96811,96813,96817)
METHODS=('shared','direct_threshold','mirror_threshold','rank2_private_synapse','independent_moe')
torch.set_num_threads(1)


def normalize_rows(w):
    return w/(np.linalg.norm(w,axis=-1,keepdims=True)+1e-12)


def make_world(seed,family):
    rng=np.random.default_rng(seed)
    if family=='shared_synapse_threshold':
        w=normalize_rows(rng.normal(size=(1,D)))[0]
        theta=np.linspace(-.8,.8,Q,dtype=np.float32)
        teachers=np.broadcast_to(w,(Q,D)).copy()
    else:
        teachers=normalize_rows(rng.normal(size=(Q,D)))
        theta=rng.uniform(-.45,.45,Q).astype(np.float32)
    splits={}
    for name,n in [('train',NTRAIN),('development',NDEV),('fresh',NTEST)]:
        xs=[];ys=[]
        for q in range(Q):
            x=rng.normal(size=(n,D)).astype(np.float32)
            y=((x@teachers[q])>theta[q]).astype(np.int64)
            xs.append(x);ys.append(y)
        splits[name]=(np.stack(xs),np.stack(ys))
    return teachers.astype(np.float32),theta,splits


class Student(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        if mode=='independent_moe':
            self.w=nn.Parameter(torch.randn(Q,D)*.1);self.theta=nn.Parameter(torch.zeros(Q))
        else:
            self.w=nn.Parameter(torch.randn(D)*.1)
            if mode=='shared': self.theta=nn.Parameter(torch.zeros(()))
            elif mode=='direct_threshold': self.theta=nn.Parameter(torch.zeros(Q))
            elif mode=='mirror_threshold':
                self.theta_base=nn.Parameter(torch.zeros(()));self.theta_scale=nn.Parameter(torch.ones(()));self.m_role=nn.Parameter(torch.zeros(Q))
            elif mode=='rank2_private_synapse':
                self.theta=nn.Parameter(torch.zeros(Q));self.role_code=nn.Parameter(torch.randn(Q,2)*.1);self.private_basis=nn.Parameter(torch.randn(2,D)*.02)
            else: raise ValueError(mode)

    def weights(self,roles):
        if self.mode=='independent_moe': return self.w[roles]
        if self.mode=='rank2_private_synapse': return self.w[None,:]+self.role_code[roles]@self.private_basis
        return self.w[None,:].expand(len(roles),-1)

    def thresholds(self,roles):
        if self.mode in ('shared','independent_moe'): return self.theta[roles] if self.mode=='independent_moe' else self.theta.expand(len(roles))
        if self.mode=='mirror_threshold': return self.theta_base+self.theta_scale*self.m_role[roles]
        return self.theta[roles]

    def logits(self,x,roles):
        # Smooth surrogate during learning; inference is hard threshold firing.
        return 5.0*(torch.sum(self.weights(roles)*x,dim=-1)-self.thresholds(roles))

    def hard_spikes(self,x,roles):
        return (self.logits(x,roles)>=0).to(torch.int64)


def pack_model(model,meta):
    buf=io.BytesIO();torch.save({'state_dict':model.state_dict(),'router':list(range(Q)),'meta':meta},buf);return buf.getvalue()


def score(model,x,y):
    xt=torch.from_numpy(x.reshape(-1,D));roles=torch.arange(Q).repeat_interleave(x.shape[1])
    yt=torch.from_numpy(y.reshape(-1))
    with torch.no_grad():
        pred=model.hard_spikes(xt,roles)
    pred_np=pred.numpy().reshape(Q,-1);truth=y
    acc=(pred_np==truth).mean(axis=1)
    return {'per_role_accuracy':acc.tolist(),'macro_accuracy':float(acc.mean()),'spike_rate':float(pred_np.mean())}


def fit(mode,seed,xtrain,ytrain,xdev,ydev):
    torch.manual_seed(seed);np.random.seed(seed)
    model=Student(mode)
    opt=torch.optim.AdamW(model.parameters(),lr=.04,weight_decay=.001)
    xt=torch.from_numpy(xtrain.reshape(-1,D));yt=torch.from_numpy(ytrain.reshape(-1).astype(np.float32))
    roles=torch.arange(Q).repeat_interleave(xtrain.shape[1])
    xd=torch.from_numpy(xdev.reshape(-1,D));yd=torch.from_numpy(ydev.reshape(-1))
    rd=torch.arange(Q).repeat_interleave(xdev.shape[1])
    best=-1.;best_state=None;best_epoch=0;stale=0;updates=0;t0=time.perf_counter()
    for epoch in range(300):
        model.train();opt.zero_grad(set_to_none=True)
        loss=nn.functional.binary_cross_entropy_with_logits(model.logits(xt,roles),yt)
        loss.backward();opt.step();updates+=1
        model.eval()
        with torch.no_grad(): acc=float((model.hard_spikes(xd,rd)==yd).float().mean())
        if acc>best:
            best=acc;best_epoch=epoch+1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
        else:
            stale+=1
            if stale>=15:break
    elapsed=time.perf_counter()-t0
    model.load_state_dict(best_state)
    return model,{'development_accuracy':best,'best_epoch':best_epoch,'optimizer_updates':updates,'wall_time_s':elapsed}


def run_world(family,seed,kind):
    teachers,theta,splits=make_world(seed,family)
    trainx,trainy=splits['train'];devx,devy=splits['development'];testx,testy=splits['fresh']
    outputs=[];models={}
    for mode in METHODS:
        model,fitm=fit(mode,seed+int(hashlib.sha1(mode.encode()).hexdigest()[:6],16)%10000,trainx,trainy,devx,devy)
        meta={'method':mode,'family':family,'world_seed':seed,'router':'identity top1','teacher':kind,'shape':[Q,D]}
        blob=pack_model(model,meta)
        reloaded=Student(mode);checkpoint=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False);reloaded.load_state_dict(checkpoint['state_dict'])
        # A trained Mirror threshold is an ordinary direct threshold vector in function space.
        fresh=score(reloaded,testx,testy);dev_replay=score(reloaded,devx,devy)
        models[mode]=reloaded
        mac_per_example=D+(2*2*D if mode=='rank2_private_synapse' else 0)
        out={'family':family,'condition':mode,'world_seed':seed,'serialized_bytes':len(blob),
             'train_examples':int(Q*NTRAIN),'optimizer_updates':fitm['optimizer_updates'],
             'active_synapse_macs':mac_per_example,'spike_rate':fresh['spike_rate'],
             'wall_time_s':fitm['wall_time_s'],'primary_metric':'fresh macro hard-spike accuracy',
             'primary_value':fresh['macro_accuracy'],'secondary_metric':'per-role accuracy; reload dev replay; best epoch',
             'secondary_value':json.dumps({'per_role':fresh['per_role_accuracy'],'dev_replay':dev_replay['macro_accuracy'],'best_epoch':fitm['best_epoch']},separators=(',',':')),
             'status_note':'role ID observed by all methods; fresh teacher world'}
        outputs.append(out)
    # Exact function-space alias: any Mirror base/scale/m code decodes to direct thresholds.
    mir=models['mirror_threshold'];direct=models['direct_threshold']
    with torch.no_grad():
        direct.theta.copy_(mir.theta_base+mir.theta_scale*mir.m_role)
        direct.w.copy_(mir.w)
    xx=torch.from_numpy(testx[:,:64].reshape(-1,D));rr=torch.arange(Q).repeat_interleave(64)
    with torch.no_grad():
        alias=float((mir.logits(xx,rr)-direct.logits(xx,rr)).abs().max())
    source={'family':family,'world_seed':seed,'teacher_weights':teachers.tolist(),'teacher_thresholds':theta.tolist(),
            'mirror_direct_max_logit_error':alias,'results':outputs}
    return outputs,source


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dev-only',action='store_true');args=parser.parse_args()
    outpath=ROOT/'RESULTS_CORE.csv';allsource=[]
    fieldnames=list(csv.DictReader(outpath.open()).fieldnames)
    # Development is run and reviewed before any fresh world is opened.
    seeds=[(96801,'development')] if args.dev_only else [(s,'fresh') for s in FRESH_SEEDS]
    for family in ('shared_synapse_threshold','generic_task_synapses'):
        for seed,kind in seeds:
            rows,detail=run_world(family,seed,kind);allsource.append(detail)
            if kind=='fresh':
                with outpath.open('a',newline='') as f: csv.DictWriter(f,fieldnames=fieldnames,lineterminator='\n').writerows(rows)
            print(json.dumps({'family':family,'world':seed,'kind':kind,'methods':[{r['condition']:r['primary_value'],'bytes':r['serialized_bytes']} for r in rows],'alias':detail['mirror_direct_max_logit_error']},sort_keys=True),flush=True)
    outfile='DEVELOPMENT_RESULTS.json' if args.dev_only else 'WORLD_RESULTS.json'
    (ROOT/'source'/outfile).write_text(json.dumps(allsource,indent=2)+'\n')

if __name__=='__main__': main()
