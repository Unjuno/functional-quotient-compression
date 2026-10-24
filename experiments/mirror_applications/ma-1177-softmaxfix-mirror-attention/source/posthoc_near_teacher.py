"""POSTHOC NON-CONFIRMATORY: near-copy transfers, DEV only.
Motivated after primary evaluation. No fresh/OOD data read and no primary gates altered.
"""
import csv,sys,math,hashlib,json,copy
from pathlib import Path
import torch
from core import *
from run import ROOT,RESULTS,load_record,model_outputs,choose_batch_order,file_sha,fp64_distill_kl

def main():
    torch.set_num_threads(1)
    manifest=json.loads((RESULTS/'SEALED_TRAIN.json').read_text())
    records={ (r['meta']['world'],r['meta']['seed'],r['meta']['kind'],r['meta']['beta'],r['meta']['loss']):r for r in manifest['records'] if r['meta']['lane']=='hard'}
    rows=[]
    for world in WORLDS:
      for seed in SEEDS:
       train=dataset(world,seed,'train',384);dev=dataset(world,seed,'dev',256)
       ids=choose_batch_order(seed,150,'distill')
       for kind in ('mirror','film'):
        teacher=load_record(records[(world,seed,kind,12.0,'fixed_ce')]);ts=teacher.state_dict()
        teacher.eval()
        t_dev=model_outputs(teacher,dev)
        for perturb in (1e-4,1e-2):
          with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed*53+int(perturb*1e6))
            delta={k:torch.randn(v.shape)*perturb for k,v in ts.items()}
          init=TinyAttention(kind,seed,12.0)
          init.load_state_dict({k:v.clone()+delta[k] for k,v in ts.items()})
          init_dev_kl=fp64_distill_kl(model_outputs(init,dev),t_dev)
          for lossname,loss in [('old',standard_soft_ce),('fixed',fixed_soft_ce)]:
            student=TinyAttention(kind,seed,12.0);student.load_state_dict(init.state_dict())
            opt=torch.optim.AdamW(student.parameters(),lr=0.02,eps=1e-8,weight_decay=0.0)
            for ix in ids:
                with torch.no_grad():tz=teacher(train['q'][ix],train['k'][ix],train['rule'][ix])
                sz=student(train['q'][ix],train['k'][ix],train['rule'][ix])
                obj=loss(sz,tz)
                opt.zero_grad(set_to_none=True);obj.backward();opt.step()
            after=model_outputs(student,dev)
            rows.append({'world':world,'seed':seed,'kind':kind,'init_noise_sigma':perturb,'loss':lossname,
               'initial_dev_teacher_student_kl':init_dev_kl,'final_dev_teacher_student_kl':fp64_distill_kl(after,t_dev),
               'max_abs_parameter_diff_to_teacher':max(float((student.state_dict()[k]-ts[k]).abs().max()) for k in ts),
               'dev_top1_mismatch':float((after.argmax(-1)!=t_dev.argmax(-1)).float().mean())})
    out=ROOT/'exploratory/NEAR_TEACHER_DEV_ONLY.csv'
    with out.open('w',newline='') as f:
       writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('POSTHOC DEV ONLY',len(rows),'rows',file_sha(out))
    from statistics import mean
    for kind in ('mirror','film'):
      for noise in (1e-4,1e-2):
        for loss in ('old','fixed'):
          sel=[r for r in rows if r['kind']==kind and r['init_noise_sigma']==noise and r['loss']==loss]
          print(kind,noise,loss,'initial',mean(r['initial_dev_teacher_student_kl'] for r in sel),'final',mean(r['final_dev_teacher_student_kl'] for r in sel),'mismatch',mean(r['dev_top1_mismatch'] for r in sel))
if __name__=='__main__':main()
