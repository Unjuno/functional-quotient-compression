"""Controlled multi-view attention teacher-code recovery.

This is an ALIGNMENT MECHANISM experiment: teacher heads are deliberately
built from the same shared cache and Mirror family. Never interpret success
as evidence that naturally trained attention heads are Mirror-compressible.
"""
import argparse
import csv
import io
import math
import os
import sys
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from multihead import SharedMultihead,code_pairwise_kl


class OffOrbitTeacher(SharedMultihead):
    """Fixed independent dense orthogonal per-head Q/output maps (off orbit)."""
    def __init__(self, world, head_count=4):
        super().__init__(d_model=16,d_head=8,heads=head_count,mode='null')
        generator=torch.Generator().manual_seed(int(world)+100004)
        q=[];v=[]
        for j in range(head_count):
            m=torch.randn((8,8),generator=generator)
            u,_=torch.linalg.qr(m)
            n=torch.randn((8,8),generator=generator)
            w,_=torch.linalg.qr(n)
            q.append(u);v.append(w)
        self.register_buffer('orth_q',torch.stack(q))
        self.register_buffer('orth_v',torch.stack(v))

    def forward(self,x,return_attention=False):
        q0,k0,v0=x@self.Wq,x@self.Wk,x@self.Wv
        length=x.shape[1]
        mask=torch.ones(length,length,dtype=torch.bool,device=x.device).triu(1)
        outs=[];att=[]
        for j in range(self.heads):
            scores=(q0@self.orth_q[j])@k0.transpose(-2,-1)/math.sqrt(self.d_head)
            a=scores.masked_fill(mask,-torch.inf).softmax(-1)
            outs.append((a@v0)@self.orth_v[j]);att.append(a)
        result=torch.stack(outs,dim=2)
        return (result,torch.stack(att,dim=2)) if return_attention else result


def make_teacher(world,head_count=4,mode='linear'):
    torch.manual_seed(world)
    if mode=='off_orbit':
        return OffOrbitTeacher(world,head_count).eval()
    teacher=SharedMultihead(d_model=16,d_head=8,heads=head_count,mode=mode)
    code_q=torch.tensor([-.82,-.33,.31,.77],dtype=torch.float32)[:head_count]
    code_v=torch.tensor([.55,-.69,.72,-.43],dtype=torch.float32)[:head_count]
    with torch.no_grad():
        teacher.q_code.copy_(code_q)
        teacher.v_code.copy_(code_v)
    teacher.eval()
    return teacher


def run_one(teacher_mode,student_mode,world,seed,updates,batch,lr,eval_n):
    teacher=make_teacher(world,mode=teacher_mode)
    torch.manual_seed(1000+seed)
    student=SharedMultihead(d_model=16,d_head=8,heads=4,mode=student_mode)
    for name in ('Wq','Wk','Wv'):
        getattr(student,name).copy_(getattr(teacher,name))
    # Same physical canonical QKV buffers, only View description is optimized.
    opt=torch.optim.Adam(student.parameters(),lr=lr) if student_mode!='null' else None
    gen=torch.Generator().manual_seed(world*113+seed*13+17)
    started=time.perf_counter()
    student.train()
    first_loss=None
    for i in range(updates):
        x=torch.randn((batch,8,16),generator=gen)
        with torch.no_grad():y=teacher(x)
        p=student(x)
        loss=F.mse_loss(p,y)
        if first_loss is None:first_loss=loss.item()
        if opt is not None:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(),2.)
            opt.step()
    elapsed=time.perf_counter()-started
    eval_gen=torch.Generator().manual_seed(world*113+seed*13+1000017)
    student.eval()
    n_seen=0
    sum_err=0
    att_kl=0.
    with torch.no_grad():
        while n_seen<eval_n:
            n=min(64,eval_n-n_seen)
            x=torch.randn((n,8,16),generator=eval_gen)
            t=teacher(x)
            o,a=student(x,return_attention=True)
            sum_err+=F.mse_loss(o,t,reduction='sum').item()
            att_kl+=n*code_pairwise_kl(a)
            n_seen+=n
    mse=sum_err/(eval_n*8*4*8)
    physical=len(student.state_dict())
    return {'teacher':teacher_mode,'student':student_mode,'world':world,'seed':seed,
        'updates':updates,'batch':batch,'eval_n':eval_n,'lr':lr,
        'initial_mse':first_loss,'heldout_mse':mse,
        'mean_pair_head_attention_kl':att_kl/eval_n,
        'code_count':sum(p.numel() for p in student.parameters()),
        'stored_model_bytes':student.serialized_bytes(),
        'train_seconds':elapsed,'dtype':'torch.float32','threads':torch.get_num_threads()}


def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--phase',choices=('dev','fresh'),required=True)
    pa.add_argument('--out',required=True)
    pa.add_argument('--updates',type=int,default=160)
    pa.add_argument('--batch',type=int,default=32)
    pa.add_argument('--lr',type=float,default=.06)
    pa.add_argument('--eval-n',type=int,default=256)
    args=pa.parse_args()
    torch.set_num_threads(1)
    worlds=[16100] if args.phase=='dev' else [16101,16102,16103]
    seeds=[1] if args.phase=='dev' else [1,2]
    out=Path(args.out)
    if out.exists(): raise FileExistsError(f'no overwriting results: {out}')
    out.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for teacher in ('linear','nonlinear','off_orbit'):
        for student in ('linear','nonlinear','gain','null'):
            for world in worlds:
                for seed in seeds:
                    row=run_one(teacher,student,world,seed,args.updates,args.batch,args.lr,args.eval_n)
                    rows.append(row)
                    with out.open('w',newline='') as fp:
                        wr=csv.DictWriter(fp,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
                    print(f'{teacher:9} {student:9} world={world} seed={seed} heldout_MSE={row["heldout_mse"]:.6g} KL={row["mean_pair_head_attention_kl"]:.5f}',flush=True)
    print('ROWS',len(rows))

if __name__=='__main__':main()
