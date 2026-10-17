"""Development-only post-audit teacher-informed Mirror direction selection."""
from __future__ import annotations
import io,csv,time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from model import N,seed_transform,nonlinear_orbit,invert_nonlinear_orbit,NativeMLP,weight_to_flat,eval_flat,MirrorMLP
from run import ROOT,RESULTS,dataset,evaluate,SEEDS,fit_weights

class AddressedMirrorMLP(nn.Module):
    def __init__(self, indices, nonlinear=True):
        super().__init__()
        self.register_buffer('indices',torch.as_tensor(indices,dtype=torch.int16))
        self.coords=nn.Parameter(torch.zeros(len(indices),dtype=torch.float64))
        self.m=len(indices)
        self.nonlinear=nonlinear
    def weights(self):
        z=torch.zeros(N,dtype=torch.float64)
        z=z.scatter(0,self.indices.long(),self.coords)
        z=seed_transform(z)
        return nonlinear_orbit(z) if self.nonlinear else z
    def forward(self,x):return eval_flat(self.weights(),x)
    def serialized_bytes(self):
        b=io.BytesIO()
        torch.save({'state':self.state_dict(),'m':self.m,'nonlinear':self.nonlinear},b)
        return len(b.getvalue())


def main():
    torch.set_num_threads(1)
    xx,yy,split=dataset()
    rows=[]
    for seed in SEEDS:
        teacher=NativeMLP()
        teacher.load_state_dict(torch.load(RESULTS/f'native_{seed}.pt',weights_only=True)['state'])
        teacher.eval()
        target=weight_to_flat(teacher)
        dev_ref,_=evaluate(teacher,xx,yy,split,'dev')
        for nonlinear in (False,True):
            inv=invert_nonlinear_orbit(target) if nonlinear else seed_transform(target,True)
            for m in (256,512,768,1024):
                for selection in ('first_m','top_abs_m'):
                    idx=(torch.arange(m) if selection=='first_m' else torch.topk(inv.abs(),m).indices.sort().values)
                    student=AddressedMirrorMLP(idx,nonlinear)
                    with torch.no_grad():student.coords.copy_(inv[idx.long()])
                    fit=fit_weights(student,target)
                    dev_acc,dev_nll=evaluate(student,xx,yy,split,'dev')
                    with torch.no_grad():err=float(torch.linalg.vector_norm(student.weights()-target))
                    row={'seed':seed,'nonlinear':nonlinear,'selection':selection,'m':m,
                         'teacher_dev_acc':dev_ref,'compiled_dev_acc':dev_acc,
                         'dev_nll':dev_nll,'weight_l2':err,
                         'serialized_bytes':student.serialized_bytes(),
                         'fit_seconds':fit['fit_seconds'],'fit_evals':fit['optimizer_iterations']}
                    rows.append(row)
                    print(seed,'mirror' if nonlinear else 'plain',selection,m,
                          'dev',round(dev_acc,4),'l2',round(err,4),flush=True)
    p=RESULTS/'direction_dev_exploratory.csv'
    with p.open('x',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

if __name__=='__main__':main()
