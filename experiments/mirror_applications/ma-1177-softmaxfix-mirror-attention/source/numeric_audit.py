"""Post-training frozen-weight numeric diagnostic, no hyperparameter selection."""
import json
import sys
import torch
from core import *
from run import ROOT,RESULTS,load_record,model_outputs


def grad(fn,z,*args):
    z=z.detach().clone().requires_grad_()
    v=fn(z,*args)
    return float(v.detach()),torch.autograd.grad(v,z)[0].detach()


def main():
    torch.set_num_threads(1)
    row=[]
    for dtype, margin in ((torch.float32,18.),(torch.bfloat16,7.),(torch.float64,40.)):
        z=torch.tensor([[margin,0.,0.]],dtype=dtype)
        t=torch.tensor([0])
        _,a=grad(standard_ce,z,t);_,b=grad(fixed_ce,z,t)
        _,c=grad(lambda x:F.softmax(x,dim=-1)[...,0].mean(),z)
        _,d=grad(lambda x:log_softmax_fixed(x).exp()[...,0].mean(),z)
        row.append({"dtype":str(dtype),"margin":margin,"CE_standard_target_gradient":float(a[0,0]),"CE_fixed_target_gradient":float(b[0,0]),"CE_standard_gradient_sum":float(a.sum()),"CE_fixed_gradient_sum":float(b.sum()),"max_softmax_standard_input_grad":float(c[0,0]),"max_softmax_fixed_input_grad":float(d[0,0])})
    manifest=json.loads((RESULTS/'SEALED_TRAIN.json').read_text())
    examples=[]
    for seed in SEEDS:
        rec=next(x for x in manifest['records'] if x['meta']['lane']=='hard' and x['meta']['world']=='householder_aligned' and x['meta']['seed']==seed and x['meta']['kind']=='mirror' and x['meta']['beta']==12. and x['meta']['loss']=='fixed_ce')
        model=load_record(rec)
        x=dataset('householder_aligned',seed,'train',384)
        logits=model_outputs(model,x)
        margins=hard_target_margin(logits,x['target'])
        ids=(margins>16.6).nonzero().flatten()
        if ids.numel()==0:
            examples.append({'seed':seed,'n_sat':0});continue
        ix=ids[margins[ids].argmax()]
        z=logits[ix:ix+1];t=x['target'][ix:ix+1]
        _,go=grad(standard_ce,z,t);_,gf=grad(fixed_ce,z,t)
        _,gd=grad(standard_ce,z.double(),t)
        examples.append({'seed':seed,'n_sat':int(ids.numel()),'total_n':384,'max_sat_margin':float(margins[ix]),'standard_target_grad':float(go[0,t.item()]),'fixed_target_grad':float(gf[0,t.item()]),'fp64_target_grad':float(gd[0,t.item()]),'standard_grad_sum':float(go.sum()),'fixed_grad_sum':float(gf.sum())})
    out={'prereg_numeric_reference':True,'dtype_cliff_tests':row,'actual_trained_Mirror_logit_test':examples,'note':'Standalone log-softmax test, not fused SDPA kernel replacement; learned-model test is from TRAIN only.'}
    (RESULTS/'NUMERIC_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':
    from torch.nn import functional as F
    main()
