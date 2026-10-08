#!/usr/bin/env python3
"""Frozen MA-461 comparison of full and code-output hypernetworks."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from pathlib import Path
import numpy as np,torch
from torch import nn
D,TASKS,LAYERS,HIDDEN=4,6,4,16;UPDATES,BATCH,NEVAL=1800,8,256
METHODS=('hyperformer','mirror','native_lowrank','shared','independent_seen')
TRAIN=[(t,l) for t in range(TASKS) for l in range(LAYERS) if (3*t+5*l)%4!=0]
HELD=[(t,l) for t in range(TASKS) for l in range(LAYERS) if (3*t+5*l)%4==0]

def teacher(seed):
    rng=np.random.default_rng(seed+461);bases=rng.normal(scale=.28,size=(2,D,D)).astype(np.float32)
    ts=np.linspace(-1,1,TASKS);ls=np.linspace(-1,1,LAYERS);targets=np.zeros((TASKS,LAYERS,D,D),np.float32)
    for t in range(TASKS):
        for l in range(LAYERS):
            a=.4*ts[t]+.25*ls[l]+.15*ts[t]*ls[l];b=.3*np.sin(np.pi*(ts[t]+ls[l])/2)
            targets[t,l]=a*bases[0]+b*bases[1]
    return torch.tensor(bases),torch.tensor(targets)

def is_train(t,l):return (t,l) in set(TRAIN)

class Generator(nn.Module):
    def __init__(self,out_dim,seed):
        super().__init__();torch.manual_seed(seed+46101)
        self.task=nn.Embedding(TASKS,2);self.layer=nn.Embedding(LAYERS,2)
        self.fc1=nn.Linear(4,HIDDEN);self.fc2=nn.Linear(HIDDEN,out_dim)
    def forward(self,t,l):return self.fc2(torch.tanh(self.fc1(torch.cat([self.task(t),self.layer(l)],dim=-1))))

def make_model(method,seed):
    if method in ('mirror','native_lowrank'):
        net=Generator(2,seed);torch.manual_seed(seed+46102);basis=nn.Parameter(torch.randn(2,D,D)*.1);return {'net':net,'basis':basis}
    if method=='hyperformer':return {'net':Generator(D*D,seed)}
    if method=='shared':
        torch.manual_seed(seed+46103);return {'delta':nn.Parameter(torch.zeros(D,D))}
    torch.manual_seed(seed+46104);return {'table':nn.Parameter(torch.zeros(TASKS,LAYERS,D,D))}

def output(method,state,t,l):
    ti=torch.tensor([t],dtype=torch.long);li=torch.tensor([l],dtype=torch.long)
    if method in ('mirror','native_lowrank'):
        c=state['net'](ti,li).reshape(2);return torch.einsum('r,rij->ij',c,state['basis'])
    if method=='hyperformer':return state['net'](ti,li).reshape(D,D)
    if method=='shared':return state['delta']
    return state['table'][t,l]

def params(method,state):
    if method in ('mirror','native_lowrank'):return list(state['net'].parameters())+[state['basis']]
    return list(state['net'].parameters()) if method=='hyperformer' else [state['delta']] if method=='shared' else [state['table']]

def train(method,seed,targets):
    state=make_model(method,seed);opt=torch.optim.Adam(params(method,state),lr=.01);rng=np.random.default_rng(seed+46111);start=time.perf_counter()
    train_pairs=TRAIN if method!='independent_seen' else TRAIN
    for _ in range(UPDATES):
        ids=rng.integers(len(train_pairs),size=BATCH);loss=0
        for j in ids:
            t,l=train_pairs[int(j)];pred=output(method,state,t,l);loss=loss+((pred-targets[t,l])**2).mean()
        loss=loss/BATCH;opt.zero_grad();loss.backward();opt.step()
    return state,time.perf_counter()-start

def pack(arr,meta):
    b=io.BytesIO();d={k:np.asarray(v) for k,v in arr.items()};d['__schema_json__']=np.frombuffer(json.dumps(meta,sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8);np.savez(b,**d);return b.getvalue()

def state_arrays(method,state):
    if method in ('mirror','native_lowrank'):
        a={f'task_embedding':state['net'].task.weight.detach().numpy(),'layer_embedding':state['net'].layer.weight.detach().numpy(),'generator_fc1_weight':state['net'].fc1.weight.detach().numpy(),'generator_fc1_bias':state['net'].fc1.bias.detach().numpy(),'generator_fc2_weight':state['net'].fc2.weight.detach().numpy(),'generator_fc2_bias':state['net'].fc2.bias.detach().numpy(),'adapter_basis':state['basis'].detach().numpy(),'base_matrix':np.eye(D,dtype=np.float32)};family='rank2-code-hypernetwork-v1'
    elif method=='hyperformer':
        n=state['net'];a={'task_embedding':n.task.weight.detach().numpy(),'layer_embedding':n.layer.weight.detach().numpy(),'generator_fc1_weight':n.fc1.weight.detach().numpy(),'generator_fc1_bias':n.fc1.bias.detach().numpy(),'generator_fc2_weight':n.fc2.weight.detach().numpy(),'generator_fc2_bias':n.fc2.bias.detach().numpy(),'base_matrix':np.eye(D,dtype=np.float32)};family='full-adapter-hypernetwork-v1'
    elif method=='shared':a={'shared_delta':state['delta'].detach().numpy(),'base_matrix':np.eye(D,dtype=np.float32)};family='shared-single-adapter-v1'
    else:a={'adapter_table':state['table'].detach().numpy(),'training_pair_mask':np.array([[is_train(t,l) for l in range(LAYERS)] for t in range(TASKS)],np.uint8),'base_matrix':np.eye(D,dtype=np.float32)};family='seen-pair-independent-table-v1'
    return a,family

def serialize(method,state):
    a,family=state_arrays(method,state);return pack(a,{'format':'MA461-hypernetwork-adapter-v1','method_family':family,'dimension':D,'tasks':TASKS,'layers':LAYERS,'dtype':'float32','training_pairs':len(TRAIN)})

def evaluate(method,state,seed,targets,train_wall):
    rng=np.random.default_rng(seed+46121);scores=[];seen=[];start=time.perf_counter()
    for t,l in HELD:
        x=torch.tensor(rng.normal(size=(NEVAL,D)).astype(np.float32));base=torch.eye(D);target=x@(base+targets[t,l]).T
        y=x@(base+output(method,state,t,l).detach()).T
        scores.append(float(torch.sqrt(((y-target)**2).mean())))
    for t,l in TRAIN:
        x=torch.tensor(rng.normal(size=(NEVAL,D)).astype(np.float32));base=torch.eye(D);target=x@(base+targets[t,l]).T;y=x@(base+output(method,state,t,l).detach()).T
        seen.append(float(torch.sqrt(((y-target)**2).mean())))
    query_wall=time.perf_counter()-start;raw=serialize(method,state)
    if method=='hyperformer':mac=4*HIDDEN+HIDDEN*HIDDEN+D*D+D*D
    elif method in ('mirror','native_lowrank'):mac=4*HIDDEN+HIDDEN*2+2*D*D+D*D+D*D
    elif method=='shared':mac=D*D
    else:mac=D*D
    qrmse=float(np.mean(scores)) if method!='independent_seen' else None
    return {'heldout_pair_rmse':scores if method!='independent_seen' else None,'mean_heldout_rmse':qrmse,'seen_pair_rmse':seen,'mean_seen_rmse':float(np.mean(seen)),'payload_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'optimizer_updates':UPDATES,'training_combinations_seen':UPDATES*BATCH,'train_wall_s':train_wall,'query_wall_s':query_wall,'inference_mac_proxy_per_input':mac,'query_examples':len(HELD)*NEVAL,'training_ops_proxy':UPDATES*BATCH*mac,'evaluation_ops_proxy':len(HELD)*NEVAL*mac,'active_ops_proxy':UPDATES*BATCH*mac+len(HELD)*NEVAL*mac},raw

def run(seed,out):
    out.mkdir(parents=True,exist_ok=True);_,targets=teacher(seed);result={};states={}
    for method in METHODS:
        state,wall=train(method,seed,targets);states[method]=state;m,raw=evaluate(method,state,seed,targets,wall);result[method]=m;(out/f'{method}_payload.npz').write_bytes(raw)
    assert result['mirror']['payload_sha256']==result['native_lowrank']['payload_sha256']
    assert result['mirror']['heldout_pair_rmse']==result['native_lowrank']['heldout_pair_rmse']
    obj={'experiment_id':'MA-461','seed':seed,'split':'dev','task':{'dimension':D,'tasks':TASKS,'layers':LAYERS,'training_pairs':len(TRAIN),'heldout_pairs':HELD,'query_examples_per_pair':NEVAL},'methods':result};(out/'metrics.json').write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');return obj

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.seed,a.out),sort_keys=True))
if __name__=='__main__':main()
