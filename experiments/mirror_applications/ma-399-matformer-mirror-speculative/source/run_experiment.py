"""Synthetic nested-width speculative decoding and residual-view comparison for MA-399."""
from __future__ import annotations
import argparse, hashlib, io, json, time, zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn

S=64; V=16; H=64; WIDTHS=(16,32,64); K=4

class SuperNet(nn.Module):
    def __init__(self):
        super().__init__(); self.w1=nn.Parameter(torch.empty(H,H)); self.b1=nn.Parameter(torch.zeros(H)); self.w2=nn.Parameter(torch.empty(V,H)); self.b2=nn.Parameter(torch.zeros(V))
        nn.init.xavier_uniform_(self.w1); nn.init.xavier_uniform_(self.w2)
    def forward(self, x, width=64):
        h=torch.nn.functional.gelu(torch.nn.functional.linear(x,self.w1[:width,:],self.b1[:width]))
        return torch.nn.functional.linear(h,self.w2[:,:width],self.b2)

def onehot(): return torch.eye(S)
def make_world(seed):
    g=torch.Generator().manual_seed(seed)
    logits=torch.randn(S,V,generator=g)*1.25
    probs=logits.softmax(-1)
    return probs

def train(seed, independent=False):
    torch.manual_seed(seed); target=make_world(seed); model=SuperNet(); opt=torch.optim.Adam(model.parameters(),lr=.003)
    start=time.perf_counter()
    for step in range(1600):
        states=torch.randint(S,(256,)); y=target[states]
        width=16 if independent else WIDTHS[step%3]
        logits=model(onehot()[states],width)
        loss=nn.functional.kl_div(logits.log_softmax(-1),y,reduction='batchmean')
        opt.zero_grad();loss.backward();opt.step()
    return model,target,time.perf_counter()-start

def fit_views(model):
    with torch.no_grad():
        residual=model(onehot(),64)-model(onehot(),16)
        u,s,v=torch.linalg.svd(residual,full_matrices=False); basis=v[:2].clone()
        coeff=residual@basis.T
        radius=coeff.norm(dim=-1).median().clamp_min(1e-6)
        theta=torch.atan2(coeff[:,1],coeff[:,0]); mirror=radius*(theta.cos()[:,None]*basis[0]+theta.sin()[:,None]*basis[1])
        scalar=coeff[:,0,None]*basis[0]
        ordinary=coeff@basis
    return basis,theta,radius,scalar,ordinary,mirror

def distributions(model,target,views):
    basis,theta,radius,scalar,ordinary,mirror=views
    l16=model(onehot(),16);l32=model(onehot(),32);l64=model(onehot(),64)
    return {'verifier64':l64.softmax(-1),'nested16':l16.softmax(-1),'routed32':l32.softmax(-1),'scalar':(l16+scalar).softmax(-1),'ordinary_rank2':(l16+ordinary).softmax(-1),'mirror':(l16+mirror).softmax(-1)}

def exact_corrected(pd,pv):
    # Accepted mass is min(p_d,p_v); rejection mass is then sampled from
    # normalize(max(p_v-p_d,0)), yielding exactly p_v after averaging.
    accepted=torch.minimum(pd,pv); residual=(pv-pd).clamp_min(0)
    reject=1-accepted.sum(-1,keepdim=True)
    correction=torch.where(reject>1e-12,residual/reject.clamp_min(1e-30),pv)
    return accepted+reject*correction

def expected_accept(pd,pv): return torch.minimum(pd,pv).sum(-1).mean().item()
def archive(path,model,view_state=None,independent=None):
    arrays={f'model.{k}':v.detach().cpu().numpy().astype(np.float16) for k,v in model.state_dict().items()}
    if view_state:
        for k,v in view_state.items(): arrays[k]=np.asarray(v,dtype=np.float16)
    if independent is not None:
        for k,v in independent.state_dict().items(): arrays[f'independent.{k}']=v.detach().cpu().numpy().astype(np.float16)
    meta=json.dumps({'experiment_id':'MA-399','state_count':S,'vocabulary':V,'hidden_width':H,'nested_widths':WIDTHS,'draft_block':K,'dtype':'float16','sampler':'categorical rejection with target residual correction'},sort_keys=True).encode()
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
        z.writestr('metadata.json',meta)
        for k,a in sorted(arrays.items()):
            b=io.BytesIO();np.save(b,a,allow_pickle=False);z.writestr(f'arrays/{k}.npy',b.getvalue())
    blob=path.read_bytes();return len(blob),hashlib.sha256(blob).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    net,target,train_wall=train(a.seed); independent,_,ind_wall=train(a.seed+100000,True)
    views=fit_views(net); dist=distributions(net,target,views)
    methods={k:v for k,v in dist.items()}
    methods['independent16']=independent(onehot(),16).softmax(-1)
    rows={}; total_start=time.perf_counter()
    for name,p in methods.items():
        tv=float((p-dist['verifier64']).abs().sum(-1).max().detach()/2)
        corrected=exact_corrected(p,dist['verifier64'])
        rows[name]={'max_tv_from_verifier':tv,'exact_correction_max_tv':float((corrected-dist['verifier64']).abs().sum(-1).max().detach()/2),'mean_kl_verifier_to_method':float((dist['verifier64']*(dist['verifier64'].clamp_min(1e-12).log()-p.clamp_min(1e-12).log())).sum(-1).mean().detach())}
        if name!='verifier64': rows[name]['expected_accepted_per_draft_block']=K*expected_accept(p,dist['verifier64'])
        state=None
        if name=='scalar': state={'residual.basis':views[0][:1].numpy(),'residual.scalar':views[3][:,0].numpy()}
        if name=='ordinary_rank2':
            with torch.no_grad(): coeff=(net(onehot(),64)-net(onehot(),16))@views[0].T
            state={'residual.basis':views[0].numpy(),'residual.coefficients':coeff.numpy()}
        if name=='mirror': state={'residual.basis':views[0].numpy(),'mirror.angle':views[1].numpy(),'mirror.radius':np.full((1,),float(views[2]))}
        path=a.out/f'{a.seed}_{name}.zip'
        if name=='independent16': size,digest=archive(path,net,state,independent)
        else: size,digest=archive(path,net,state)
        rows[name].update({'actual_payload_bytes':size,'payload_sha256':digest})
    rows['benchmark_wall_seconds']=time.perf_counter()-total_start
    result={'experiment_id':'MA-399','seed':a.seed,'train_updates':1600,'train_examples_seen':409600,'supernet_train_wall_seconds':train_wall,'independent_draft_train_wall_seconds':ind_wall,'methods':rows,'target_context_examples':S,'note':'Distribution and Python CPU throughput are small synthetic proxies; no fresh worlds opened.'}
    a.out.mkdir(parents=True,exist_ok=True);(a.out/f'{a.seed}_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
