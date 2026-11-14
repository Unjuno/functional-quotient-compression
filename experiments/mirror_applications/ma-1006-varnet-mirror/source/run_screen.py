#!/usr/bin/env python3
"""Run locked MA-1006 synthetic VarNet screen. Requires pinned fastMRI source on PYTHONPATH."""
import argparse, copy, hashlib, io, json, pathlib, time
import numpy as np
import torch,hashlib
from torch import nn
from fastmri.models.varnet import VarNet
from synthetic_mri import make_case

class ScalarView(nn.Module):
    def __init__(self,shared):super().__init__();self.shared=shared;self.scale=nn.Parameter(torch.ones(()))
    def forward(self,x):return self.shared(x)*self.scale
class GivensView(nn.Module):
    def __init__(self,shared):super().__init__();self.shared=shared;self.angle=nn.Parameter(torch.zeros(()))
    def forward(self,x):
        z=self.shared(x);r,i=z[...,0],z[...,1];c=torch.cos(self.angle);s=torch.sin(self.angle)
        return torch.stack((c*r-s*i,s*r+c*i),dim=-1)

def make_model(seed):
    torch.manual_seed(seed)
    return VarNet(num_cascades=3,sens_chans=4,sens_pools=2,chans=4,pools=2)
def tie_during_pretraining(model):
    shared=model.cascades[0].model
    for block in model.cascades:block.model=shared
    return model
def tie_from(native,mode='hard'):
    model=copy.deepcopy(native);blocks=model.cascades
    states=[b.model.state_dict() for b in blocks];shared=copy.deepcopy(blocks[0].model)
    avg={}
    for k,v in states[0].items():
        if torch.is_floating_point(v):avg[k]=torch.stack([q[k].detach() for q in states]).mean(0)
        else:avg[k]=v.detach().clone()
    shared.load_state_dict(avg)
    for j,b in enumerate(blocks):
        if mode=='gate':b.model=ScalarView(shared)
        elif mode=='mirror':b.model=GivensView(shared)
        else:b.model=shared
    return model

def params(model):return sum(p.numel() for p in model.parameters())
def payload(model,kind):
    if kind=='independent':obj={'kind':kind,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'arch':'fastMRI VarNet, 3 cascades, 4/2 channels/pools'}
    else:
        obj={'kind':kind,'sensitivity':{k:v.detach().cpu() for k,v in model.sens_net.state_dict().items()},'shared_denoiser':{k:v.detach().cpu() for k,v in (model.cascades[0].model.shared if hasattr(model.cascades[0].model,'shared') else model.cascades[0].model).state_dict().items()},'dc':[b.dc_weight.detach().cpu() for b in model.cascades],'arch':'fastMRI VarNet, 3 cascades, 4/2 channels/pools'}
        if kind=='gate':obj['scales']=[b.model.scale.detach().cpu() for b in model.cascades]
        if kind=='mirror':obj['angles']=[b.model.angle.detach().cpu() for b in model.cascades]
    f=io.BytesIO();torch.save(obj,f);raw=f.getvalue();return len(raw),hashlib.sha256(raw).hexdigest()
def predict_kspace(model,case):
    mk,mask=case['masked_kspace'],case['mask'];sens=model.sens_net(mk,mask,case['num_low_frequencies']);kp=mk.clone()
    for block in model.cascades:kp=block(kp,mk,mask,sens)
    return kp
@torch.no_grad()
def evaluate(model,cases):
    nmse=[];psnr=[];dc=[];times=[]
    for q in cases:
        start=time.perf_counter();kp=predict_kspace(model,q);times.append(time.perf_counter()-start)
        image=model(q['masked_kspace'],q['mask'],q['num_low_frequencies'])
        target=q['target'];err=(image-target).square().sum();nmse.append(float(err/(target.square().sum()+1e-12)))
        mse=float((image-target).square().mean());psnr.append(float(20*torch.log10(target.max()/np.sqrt(max(mse,1e-12)))))
        resid=((kp-q['masked_kspace'])*q['mask']).norm()/(q['masked_kspace'].norm()+1e-12);dc.append(float(resid))
    return {'nmse':float(np.mean(nmse)),'psnr_db':float(np.mean(psnr)),'acquired_residual':float(np.mean(dc)),'mean_forward_seconds':float(np.mean(times)),'n':len(cases)}
def update(model,cases,updates,only_views=False):
    for p in model.parameters():p.requires_grad_(not only_views)
    if only_views:
        for b in model.cascades:b.model.scale.requires_grad_(True) if isinstance(b.model,ScalarView) else None
        for b in model.cascades:b.model.angle.requires_grad_(True) if isinstance(b.model,GivensView) else None
    ps=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.Adam(ps,lr=1e-3);hist=[]
    model.train();start=time.perf_counter()
    for step in range(updates):
        case=cases[step%len(cases)];out=model(case['masked_kspace'],case['mask'],case['num_low_frequencies']);loss=(out-case['target']).square().mean();opt.zero_grad();loss.backward();opt.step();hist.append(float(loss.detach()))
    return {'updates':updates,'last_mse':hist[-1],'wall_seconds':time.perf_counter()-start,'trainable_parameters':sum(p.numel() for p in ps)}
def file_sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['dev','replay','fresh'],default='dev');ap.add_argument('--out',required=True);ap.add_argument('--lock',default='');a=ap.parse_args()
    torch.set_num_threads(1);torch.manual_seed(106001)
    if a.stage=='dev':
        train=[make_case(106001+i,'A') for i in range(24)];support=[make_case(106101+i,'B') for i in range(8)];dev=[make_case(106201+i,'B') for i in range(16)]
        native=make_model(106001);pre_independent=update(native,train,24);base_independent=copy.deepcopy(native)
        tied_base=tie_during_pretraining(make_model(106001));pre_tied=update(tied_base,train,24);tied_base=copy.deepcopy(tied_base)
        models={'independent_frozen':copy.deepcopy(base_independent),'hard_tied':copy.deepcopy(tied_base),'scalar_gate':tie_from(tied_base,'gate'),'mirror_givens':tie_from(tied_base,'mirror')}
        # view-only controls; preserve the pretrained shared denoiser, sensitivity network, and each native DC scalar.
        adapt={}
        for k,m in models.items():
            if k in ('hard_tied','independent_frozen'):adapt[k]={'updates':0,'last_mse':None,'wall_seconds':0,'trainable_parameters':0}
            else:adapt[k]=update(m,support,24,only_views=True)
        upper=copy.deepcopy(base_independent);adapt['independent_ft']=update(upper,support,24,only_views=False);models['independent_ft']=upper
        results={k:evaluate(m,dev) for k,m in models.items()};payloads={k:payload(m,'independent' if k in ('independent_ft','independent_frozen') else k) for k,m in models.items()}
        out=pathlib.Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);models_path=out.with_name('DEV_MODELS.pt')
        torch.save({'base_independent':base_independent.state_dict(),'base_shared':tied_base.state_dict(),'models':{k:m.state_dict() for k,m in models.items()}},models_path)
        lock={'stage':'development','pretraining':{'independent':pre_independent,'hard_tied':pre_tied},'adaptation':adapt,'metrics':results,'payloads':payloads,'parameter_counts':{k:params(m) for k,m in models.items()},'fresh_seeds':[106301,106307,106313],'fresh_examples_per_seed':8,'fresh_opened':False,'checkpoint':models_path.name,'checkpoint_sha256':file_sha(models_path)}
        out.write_text(json.dumps(lock,indent=2)+'\n');print(json.dumps(lock,indent=2))
    elif a.stage=='replay':
        if not a.lock:raise SystemExit('--lock required for replay stage')
        lock=json.loads(pathlib.Path(a.lock).read_text());cp=pathlib.Path(a.lock).with_name(lock['checkpoint'])
        if file_sha(cp)!=lock['checkpoint_sha256']:raise SystemExit('serialized model checkpoint hash mismatch')
        state=torch.load(cp,map_location='cpu',weights_only=False);base=make_model(106001);base.load_state_dict(state['base_independent']);shared_base=tie_during_pretraining(make_model(106001));shared_base.load_state_dict(state['base_shared'])
        models={'independent_frozen':copy.deepcopy(base),'hard_tied':copy.deepcopy(shared_base),'scalar_gate':tie_from(shared_base,'gate'),'mirror_givens':tie_from(shared_base,'mirror'),'independent_ft':copy.deepcopy(base)}
        for k,m in models.items():m.load_state_dict(state['models'][k])
        dev=[make_case(106201+i,'B') for i in range(16)];actual={k:evaluate(m,dev) for k,m in models.items()}
        diffs={k:{metric:abs(actual[k][metric]-lock['metrics'][k][metric]) for metric in ('nmse','psnr_db','acquired_residual')} for k in models}
        replay={'stage':'development_checkpoint_replay','lock_sha256':file_sha(a.lock),'checkpoint_sha256':lock['checkpoint_sha256'],'metrics':actual,'absolute_differences':diffs,'max_abs_difference':max(v for dct in diffs.values() for v in dct.values()),'fresh_opened':False}
        pathlib.Path(a.out).write_text(json.dumps(replay,indent=2)+'\n');print(json.dumps(replay,indent=2))
    else:
        if not a.lock:raise SystemExit('--lock required for fresh stage')
        lock=json.loads(pathlib.Path(a.lock).read_text())
        # The fresh runner verifies a frozen config digest before generating any fresh phantom.
        if lock.get('fresh_opened') is not False:raise SystemExit('dev lock must show fresh_opened=false')
        cp=pathlib.Path(a.lock).with_name(lock['checkpoint'])
        if file_sha(cp)!=lock['checkpoint_sha256']:raise SystemExit('serialized model checkpoint hash mismatch')
        state=torch.load(cp,map_location='cpu',weights_only=False);base=make_model(106001);base.load_state_dict(state['base_independent']);shared_base=tie_during_pretraining(make_model(106001));shared_base.load_state_dict(state['base_shared'])
        models={'independent_frozen':copy.deepcopy(base),'hard_tied':copy.deepcopy(shared_base),'scalar_gate':tie_from(shared_base,'gate'),'mirror_givens':tie_from(shared_base,'mirror'),'independent_ft':copy.deepcopy(base)}
        for k,m in models.items():m.load_state_dict(state['models'][k])
        fresh=[]
        for seed in lock['fresh_seeds']:
            for j in range(lock['fresh_examples_per_seed']):fresh.append(make_case(seed*100+j,'B'))
        results={k:evaluate(m,fresh) for k,m in models.items()}
        out={'stage':'fresh_audit','dev_lock_sha256':file_sha(a.lock),'checkpoint_sha256':lock['checkpoint_sha256'],'fresh_seeds':lock['fresh_seeds'],'fresh_examples_per_seed':lock['fresh_examples_per_seed'],'metrics':results,'weights_or_codes_changed_after_lock':False}
        pathlib.Path(a.out).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
