"""EXPLORATORY AFTER THE PRIMARY FRESH ANALYSIS.
Ablation: does giving the shared student the original pretrained MLP core,
while withholding rule-specific codes, make functional distillation easier?
This is NOT included in the frozen MA-1178 primary pass/fail gates.
"""
from __future__ import annotations
import csv,hashlib,torch,sys,time,json
from pathlib import Path
from torch.nn import functional as F
from model import DenseMLP,new_student,teacher_mirror_family,teacher_film_family,param_count
from data import student_sets,fresh_sets
from run import SEEDS,BATCH,LR,TEACHER_WIDTH,STEPS,teacher_outputs

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'exploratory';OUT.mkdir(exist_ok=True)
torch.set_num_threads(1)
rows=[]
for world in ['mirror_related','film_related']:
 for seed in SEEDS:
  dat=torch.load(ROOT/'checkpoints'/f'teacher_{seed}.pt',map_location='cpu',weights_only=True)
  source=DenseMLP(16);source.load_state_dict(dat['base']);source.eval().requires_grad_(False)
  teachers=teacher_mirror_family(source,dat['mirror_dirs']) if world=='mirror_related' else teacher_film_family(source,dat['film_codes'])
  tr,dev,norm=student_sets(world,seed,teachers)
  fresh,ood=fresh_sets(world,seed,teachers)
  x,r,y=tr;dx,dr,dy=dev;fx,fr,fy=fresh;ox,orr,oy=ood
  world_id=['mirror_related','film_related'].index(world)
  g=torch.Generator().manual_seed(2_000_000+seed*100+['mirror_related','film_related','independent'].index(world)*1000)
  batches=torch.randint(len(x),(STEPS,BATCH),generator=g)
  for kind in ['mirror','film']:
   for core_trainable in [False,True]:
    student=new_student(kind,16,720_000+seed*13+world_id*733+16*179)
    student.up.load_state_dict(source.up.state_dict());student.down.load_state_dict(source.down.state_dict())
    for p in student.up.parameters():p.requires_grad=core_trainable
    for p in student.down.parameters():p.requires_grad=core_trainable
    optim=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],lr=LR,weight_decay=1e-4)
    with torch.inference_mode():initial=((student(dx,dr)-dy).square().mean()/norm).item()
    best=1e9;best_at=None;state=None;tic=time.perf_counter()
    for step in range(1,STEPS+1):
     batch=batches[step-1];loss=((student(x[batch],r[batch])-y[batch]).square().mean()/norm)
     optim.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad],1.0);optim.step()
     if step in [250,500,750]:
      with torch.inference_mode():val=((student(dx,dr)-dy).square().mean()/norm).item()
      if val<best:best=val;best_at=step;state={key:v.detach().clone() for key,v in student.state_dict().items()}
    elapsed=time.perf_counter()-tic
    student.load_state_dict(state)
    with torch.inference_mode():
     fresh_nmse=((student(fx,fr)-fy).square().mean()/norm).item()
     ood_nmse=((student(ox,orr)-oy).square().mean()/norm).item()
    row={'world':world,'seed':seed,'kind':kind,'transfer_core_trainable':core_trainable,'trainable_scalar_count':sum(p.numel() for p in student.parameters() if p.requires_grad),'total_scalar_count':param_count(student),'initial_dev_nmse':initial,'best_dev_nmse':best,'best_step':best_at,'fresh_nmse':fresh_nmse,'ood_nmse':ood_nmse,'train_wall_s':elapsed}
    rows.append(row)
  print('posthoc transfer:',world,seed,'models',len(rows),flush=True)
with (OUT/'TRANSFER_RESULTS.csv').open('w',newline='') as f:
 wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
(OUT/'STATUS.json').write_text(json.dumps({'status':'EXPLORATORY_AFTER_PRIMARY_FRESH','description':'Source MLP core transferred; codes randomly initialized; student does not receive true code vectors. Additional teacher parameter information means it is NOT comparable to random-init primary arms without qualification. All 2 worlds x5 seeds x2 methods x2 core regimes reported.','n_models':len(rows),'rows':rows},indent=2)+'\n')
