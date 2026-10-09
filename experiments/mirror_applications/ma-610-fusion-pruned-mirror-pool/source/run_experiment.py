#!/usr/bin/env python3
"""MA-610 usage-guided adapter pool pruning versus random/small-from-start."""
import argparse,hashlib,io,json,platform,time
from pathlib import Path
import torch
from torch import nn
D,O,R,K=12,10,2,8

def rotate(z,theta):
    c,s=theta.cos(),theta.sin();return torch.stack((c*z[:,0]-s*z[:,1],s*z[:,0]+c*z[:,1]),-1)
class Teacher:
    def __init__(self,seed):
        g=torch.Generator().manual_seed(seed);self.a=torch.randn(R,D,generator=g)*.18;self.b=torch.randn(O,R,generator=g)*.18;self.angles=torch.randn(4,generator=g)*.7;self.gate=nn.Linear(D,4)
        with torch.no_grad():self.gate.weight.copy_(torch.randn(4,D,generator=g)*.45);self.gate.bias.copy_(torch.randn(4,generator=g)*.15)
    def outputs(self,x):
        z=x@self.a.T;return torch.stack([rotate(z,self.angles[i])@self.b.T for i in range(4)],1)
    def __call__(self,x):return torch.einsum('nk,nko->no',torch.softmax(self.gate(x),-1),self.outputs(x))
class Student(nn.Module):
    def __init__(self,mode,k,seed):
        super().__init__();torch.manual_seed(seed);self.mode=mode;self.k=k;self.router=nn.Linear(D,k)
        if mode=='mirror':self.a=nn.Parameter(torch.randn(R,D)*.15);self.b=nn.Parameter(torch.randn(O,R)*.15);self.angles=nn.Parameter(torch.randn(k)*.3)
        else:self.a=nn.Parameter(torch.randn(k,R,D)*.15);self.b=nn.Parameter(torch.randn(k,O,R)*.15)
    def outputs(self,x):
        if self.mode=='mirror':
            z=x@self.a.T;return torch.stack([rotate(z,self.angles[i])@self.b.T for i in range(self.k)],1)
        return torch.stack([(x@self.a[i].T)@self.b[i].T for i in range(self.k)],1)
    def forward(self,x):return torch.einsum('nk,nko->no',torch.softmax(self.router(x),-1),self.outputs(x))
def make_data(seed,t,n):
    g=torch.Generator().manual_seed(seed);x=torch.randn(n,D,generator=g)
    with torch.no_grad():y=t(x)
    return x,y
def fit(mode,k,world,t,train,steps,tag):
    model=Student(mode,k,world+tag);opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=0);x,y=train;g=torch.Generator().manual_seed(world+tag+9000);start=time.perf_counter()
    for _ in range(steps):
        ix=torch.randint(x.shape[0],(128,),generator=g);loss=((model(x[ix])-y[ix])**2).mean();opt.zero_grad();loss.backward();opt.step()
    return model,time.perf_counter()-start
def nrmse(model,x,y):
    with torch.no_grad():return (((model(x)-y)**2).mean()/y.var()).item()
def get_state(model,indices=None):
    if indices is None:return {k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
    idx=torch.as_tensor(indices,dtype=torch.long)
    if model.mode=='mirror':
        return {'a':model.a.detach().cpu().contiguous(),'b':model.b.detach().cpu().contiguous(),'angles':model.angles[idx].detach().cpu().contiguous(),'router.weight':model.router.weight[idx].detach().cpu().contiguous(),'router.bias':model.router.bias[idx].detach().cpu().contiguous(),'kept_indices':idx}
    return {'a':model.a[idx].detach().cpu().contiguous(),'b':model.b[idx].detach().cpu().contiguous(),'router.weight':model.router.weight[idx].detach().cpu().contiguous(),'router.bias':model.router.bias[idx].detach().cpu().contiguous(),'kept_indices':idx}
def predict_pruned(mode,state,x):
    logits=x@state['router.weight'].T+state['router.bias'];z=x@state['a'].T if mode=='mirror' else None
    ys=[]
    for i in range(state['router.weight'].shape[0]):
        if mode=='mirror':v=rotate(z,state['angles'][i])@state['b'].T
        else:v=(x@state['a'][i].T)@state['b'][i].T
        ys.append(v)
    return torch.einsum('nk,nko->no',torch.softmax(logits,-1),torch.stack(ys,1))
def pack(mode,state,seed,label):
    b=io.BytesIO();torch.save({'state':state,'mode':mode,'dims':[D,O,R,state['router.weight'].shape[0]],'seed':seed,'label':label,'format':'MA610-v1'},b);return b.getvalue()
def proxy(mode,k):return D*k+k*(D*R+R*O)+(4*k if mode=='mirror' else 0)
def run_world(world,steps):
    t=Teacher(world);tr=make_data(world+100,t,4096);dv=make_data(world+200,t,1024);out={};root=Path(__file__).resolve().parents[1]/'runs/dev_payloads';root.mkdir(parents=True,exist_ok=True)
    for mode in ('mirror','full'):
        big,bigt=fit(mode,K,world,t,tr,steps,2000+(0 if mode=='mirror' else 100));small,smallt=fit(mode,4,world,t,tr,steps,3000+(0 if mode=='mirror' else 100))
        with torch.no_grad():usage=torch.softmax(big.router(tr[0]),-1).mean(0)
        top=torch.topk(usage,4).indices.tolist();g=torch.Generator().manual_seed(world+5000+(0 if mode=='mirror' else 50));randoms=[]
        for j in range(10):randoms.append(torch.randperm(K,generator=g)[:4].tolist())
        rows={};
        for label,indices,model in [('overcomplete',None,big),('small_from_start',None,small)]:
            state=get_state(model,indices);blob=pack(mode,state,world,label);path=root/f'{world}_{mode}_{label}.pt';path.write_bytes(blob)
            loaded=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)['state'];pred=predict_pruned(mode,loaded,dv[0])
            with torch.no_grad():ref=model(dv[0]);err=(pred-ref).abs().max().item()
            rows[label]={'nrmse2':(((pred-dv[1])**2).mean()/dv[1].var()).item(),'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'file':str(path.relative_to(Path(__file__).resolve().parents[1])),'compute_proxy':proxy(mode,K if label=='overcomplete' else 4),'wall_clock_seconds':bigt if label=='overcomplete' else smallt,'replay_max_abs_error':err}
        usage_state=get_state(big,top);usage_blob=pack(mode,usage_state,world,'usage_pruned');up=root/f'{world}_{mode}_usage_pruned.pt';up.write_bytes(usage_blob)
        replay_state=torch.load(io.BytesIO(usage_blob),map_location='cpu',weights_only=False)['state'];usage_pred=predict_pruned(mode,replay_state,dv[0]);usage_err=(((usage_pred-dv[1])**2).mean()/dv[1].var()).item();usage_replay=(usage_pred-predict_pruned(mode,usage_state,dv[0])).abs().max().item()
        rows['usage_pruned']={'nrmse2':usage_err,'bytes':len(usage_blob),'sha256':hashlib.sha256(usage_blob).hexdigest(),'file':str(up.relative_to(Path(__file__).resolve().parents[1])),'kept_indices':top,'usage_scores':[float(usage[i]) for i in top],'compute_proxy':proxy(mode,4),'wall_clock_seconds':bigt,'replay_max_abs_error':usage_replay}
        rand_losses=[]
        for j,ids in enumerate(randoms):
            st=get_state(big,ids);rand_losses.append(nrmse_pruned(mode,st,dv[0],dv[1]))
            if j==0:
                blob=pack(mode,st,world,'random_pruned');rp=root/f'{world}_{mode}_random_pruned.pt';rp.write_bytes(blob);loaded=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)['state'];replay=nrmse_pruned(mode,loaded,dv[0],dv[1]);rows['random_pruned_payload']={'nrmse2':replay,'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'file':str(rp.relative_to(Path(__file__).resolve().parents[1])),'kept_indices':ids,'compute_proxy':proxy(mode,4),'wall_clock_seconds':bigt,'replay_max_abs_error':(predict_pruned(mode,loaded,dv[0])-predict_pruned(mode,st,dv[0])).abs().max().item()}
        rows['random_prune_mean']={'nrmse2':sum(rand_losses)/len(rand_losses),'std':float(torch.tensor(rand_losses).std(unbiased=False)),'masks':len(rand_losses)}
        rows['training']={'overcomplete_seconds':bigt,'small_from_start_seconds':smallt,'updates_per_run':steps,'train_examples':4096,'dev_examples':1024}
        out[mode]=rows
    return out

def nrmse_pruned(mode,state,x,y):
    with torch.no_grad():return (((predict_pruned(mode,state,x)-y)**2).mean()/y.var()).item()
def main():
    p=argparse.ArgumentParser();p.add_argument('--worlds',type=int,nargs='+',default=[61001,61002]);p.add_argument('--steps',type=int,default=1200);p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(1)
    z={'experiment_id':'MA-610','phase':'development','python':platform.python_version(),'torch':torch.__version__,'device':'cpu','steps':a.steps,'worlds':{str(w):run_world(w,a.steps) for w in a.worlds}}
    o=Path(a.out);o.parent.mkdir(parents=True,exist_ok=True);o.write_text(json.dumps(z,indent=2)+'\n');print(json.dumps(z,indent=2))
if __name__=='__main__':main()
