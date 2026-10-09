#!/usr/bin/env python3
"""Frozen MA-471 shared rank-one template and per-edit Givens angle screen."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from pathlib import Path
import numpy as np,torch
from torch import nn
D,EDITS,TRAIN_EDITS=24,20,16
SUPPORT,QUERY,NLOCAL=256,128,512
UPDATES,BATCH=3000,8
RADIUS=1.8
METHODS=('mirror','native_givens','rome_factors','independent_fit','no_edit')

def rotate(v,theta):
    c=torch.cos(theta);s=torch.sin(theta);out=v.clone()
    out[0]=c*v[0]-s*v[1];out[1]=s*v[0]+c*v[1]
    return out

def delta_from_angle(theta,u,v):return torch.outer(rotate(u,theta),rotate(v,-theta))

def world(seed):
    rng=np.random.default_rng(seed+471)
    base=torch.tensor(rng.normal(scale=.2,size=(D,D)).astype(np.float32))
    u=torch.tensor(rng.normal(scale=.2,size=D).astype(np.float32));v=torch.tensor(rng.normal(scale=.2,size=D).astype(np.float32))
    angles=torch.tensor(rng.uniform(-.6,.6,size=EDITS).astype(np.float32))
    keys=torch.tensor(np.eye(D,dtype=np.float32)[:EDITS]);xs=[];ys=[];signals=[];xq=[];yq=[];deltas=[]
    for e in range(EDITS):
        delta=delta_from_angle(angles[e],u,v);deltas.append(delta)
        rs=np.random.default_rng(seed+5000+e)
        x=torch.tensor(keys[e].numpy()+rs.normal(scale=.28,size=(SUPPORT,D)).astype(np.float32))
        y=x@(base+delta).T+torch.tensor(rs.normal(scale=.01,size=(SUPPORT,D)).astype(np.float32))
        signals.append((y-x@base.T).T@x/SUPPORT);xs.append(x);ys.append(y)
        rq=np.random.default_rng(seed+70000+e)
        xv=torch.tensor(keys[e].numpy()+rq.normal(scale=.18,size=(QUERY,D)).astype(np.float32))
        xq.append(xv);yq.append(xv@(base+delta).T)
    xl=torch.tensor(np.random.default_rng(seed+90000).normal(size=(NLOCAL,D)).astype(np.float32))
    return {'base':base,'u':u,'v':v,'angles':angles,'keys':keys,'deltas':torch.stack(deltas),'signals':torch.stack(signals),'xs':xs,'ys':ys,'xq':xq,'yq':yq,'xl':xl}

class AngleEditor(nn.Module):
    def __init__(self,seed):
        super().__init__();torch.manual_seed(seed+47101);self.fc1=nn.Linear(D*D,32);self.fc2=nn.Linear(32,1)
    def forward(self,g):return .75*torch.tanh(self.fc2(torch.tanh(self.fc1(g.reshape(-1)))).reshape(()))

def init(method,seed):
    if method in ('mirror','native_givens'):
        net=AngleEditor(seed);torch.manual_seed(seed+47102)
        u=nn.Parameter(torch.randn(D)*.2);v=nn.Parameter(torch.randn(D)*.2)
        return {'net':net,'u':u,'v':v}
    return {}

def train(method,seed,w):
    if method not in ('mirror','native_givens'):return None,0.
    s=init(method,seed);opt=torch.optim.Adam(list(s['net'].parameters())+[s['u'],s['v']],lr=.005);rng=np.random.default_rng(seed+47111);start=time.perf_counter()
    for _ in range(UPDATES):
        es=rng.integers(TRAIN_EDITS,size=BATCH);loss=0.
        for e in es:
            theta=s['net'](w['signals'][e]);pred=delta_from_angle(theta,s['u'],s['v']);loss=loss+((pred-w['deltas'][e])**2).mean()
        loss=loss/BATCH;opt.zero_grad();loss.backward();opt.step()
    return {k:(v.detach() if isinstance(v,torch.Tensor) else v) for k,v in s.items()},time.perf_counter()-start

def route(x,keys):
    d=((keys-x.unsqueeze(0))**2).sum(-1);i=int(torch.argmin(d).item());return i,float(d[i])<=RADIUS*RADIUS

def support_controls(w):
    rome=[];ind=[]
    for x,y in zip(w['xs'],w['ys']):
        res=y-x@w['base'].T;full=torch.linalg.lstsq(x,res).solution.T.detach();ind.append(full)
        U,S,Vh=torch.linalg.svd(full,full_matrices=False);root=torch.sqrt(torch.clamp(S[0],min=0));rome.append((U[:,0]*root,Vh[0]*root))
    return rome,torch.stack(ind)

def pack(arr,meta):
    b=io.BytesIO();d={k:np.asarray(v) for k,v in arr.items()};d['__schema_json__']=np.frombuffer(json.dumps(meta,sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8);np.savez(b,**d);return b.getvalue()

def serialize_bank(method,w,updates,angles=None,u=None,v=None):
    obj={'base':w['base'].numpy(),'edit_keys':w['keys'].numpy(),'route_radius':np.asarray([RADIUS],np.float32)}
    if method in ('mirror','native_givens'):
        obj.update({'template_left':u.numpy(),'template_right':v.numpy(),'edit_angles':angles.numpy()});family='shared-givens-rank1-bank-v1'
    elif method=='rome_factors':obj.update({'rome_left':np.stack([a.numpy() for a,b in updates]),'rome_right':np.stack([b.numpy() for a,b in updates])});family='rome-rank1-factor-bank-v1'
    elif method=='independent_fit':obj.update({'independent_updates':updates.numpy()});family='independent-full-update-bank-v1'
    else:family='frozen-no-edit-bank-v1'
    return pack(obj,{'format':'MA471-local-rank1-bank-v1','method_family':family,'dimension':D,'edits':EDITS,'dtype':'float32'})

def serialize_editor(s):
    return pack({'fc1_weight':s['net'].fc1.weight.detach().numpy(),'fc1_bias':s['net'].fc1.bias.detach().numpy(),'fc2_weight':s['net'].fc2.weight.detach().numpy(),'fc2_bias':s['net'].fc2.bias.detach().numpy()},{'format':'MA471-angle-editor-v1','dtype':'float32'})

def evaluate(method,s,seed,w,train_wall,rome,ind):
    genstart=time.perf_counter();updates=[];angles=[]
    with torch.no_grad():
        if method in ('mirror','native_givens'):
            for e in range(EDITS):
                a=s['net'](w['signals'][e]);angles.append(a);updates.append(delta_from_angle(a,s['u'],s['v']))
        elif method=='rome_factors':
            for l,r in rome:updates.append(torch.outer(l,r))
        elif method=='independent_fit':updates=list(ind)
        else:updates=[torch.zeros(D,D) for _ in range(EDITS)]
    angles_t=torch.stack(angles) if angles else None;updates_t=torch.stack(updates);generation_wall=time.perf_counter()-genstart
    qstart=time.perf_counter();edit=[];route_ok=[];local=[];localroute=[]
    with torch.no_grad():
        for e in range(TRAIN_EDITS,EDITS):
            for x,y in zip(w['xq'][e],w['yq'][e]):
                i,on=route(x,w['keys']);route_ok.append(i==e and on);delta=updates_t[i] if on else torch.zeros(D,D)
                edit.append(float(torch.mean((x@(w['base']+delta).T-y)**2)))
        for x in w['xl']:
            i,on=route(x,w['keys']);localroute.append(on);delta=updates_t[i] if on else torch.zeros(D,D)
            local.append(float(torch.mean((x@(w['base']+delta).T-x@w['base'].T)**2)))
    query_wall=time.perf_counter()-qstart
    if method in ('mirror','native_givens'):bank=serialize_bank(method,w,updates_t,angles_t,s['u'],s['v']);editor=serialize_editor(s)
    else:bank=serialize_bank(method,w,rome if method=='rome_factors' else updates_t);editor=b''
    edit_mac=D*D*32+32+96 if method in ('mirror','native_givens') else (D*D if method=='independent_fit' else 2*D)
    route_ops=EDITS*D*3;query_mac=D*D+2*D
    metrics={'heldout_edit_rmse':float(np.sqrt(np.mean(edit))),'per_query_edit_mse':edit,'locality_rmse':float(np.sqrt(np.mean(local))),'wrong_route_rate':1-float(np.mean(route_ok)),'locality_route_rate':float(np.mean(localroute)),'inference_payload_bytes':len(bank),'editor_payload_bytes':len(editor),'full_online_edit_system_bytes':len(bank)+len(editor),'payload_sha256':hashlib.sha256(bank).hexdigest(),'editor_sha256':hashlib.sha256(editor).hexdigest() if editor else None,'optimizer_updates':UPDATES if method in ('mirror','native_givens') else 0,'support_examples_seen':UPDATES*BATCH if method in ('mirror','native_givens') else EDITS*SUPPORT,'training_wall_s':train_wall,'edit_generation_wall_s':generation_wall,'query_wall_s':query_wall,'edit_generation_ops_proxy':EDITS*edit_mac,'query_ops_proxy_per_example':query_mac+route_ops,'route_comparisons_per_example':EDITS-1,'route_distance_scalar_ops_per_example':route_ops,'active_ops_proxy':(UPDATES*BATCH*edit_mac*3 if method in ('mirror','native_givens') else EDITS*SUPPORT*D*D)+EDITS*edit_mac+((EDITS-TRAIN_EDITS)*QUERY+NLOCAL)*(query_mac+route_ops)}
    return metrics,bank,editor

def run(seed,out):
    out.mkdir(parents=True,exist_ok=True);w=world(seed);rome,ind=support_controls(w);result={}
    for method in METHODS:
        s,wall=train(method,seed,w);m,b,e=evaluate(method,s,seed,w,wall,rome,ind);result[method]=m;(out/f'{method}_bank.npz').write_bytes(b)
        if e:(out/f'{method}_editor.npz').write_bytes(e)
    assert (out/'mirror_bank.npz').read_bytes()==(out/'native_givens_bank.npz').read_bytes()
    assert result['mirror']['heldout_edit_rmse']==result['native_givens']['heldout_edit_rmse']
    doc={'experiment_id':'MA-471','seed':seed,'split':'dev','task':{'dimension':D,'edits':EDITS,'train_edits':TRAIN_EDITS,'heldout_edits':EDITS-TRAIN_EDITS,'support_per_edit':SUPPORT,'route_radius':RADIUS},'methods':result};(out/'metrics.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');return doc

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.seed,a.out),sort_keys=True))
if __name__=='__main__':main()
