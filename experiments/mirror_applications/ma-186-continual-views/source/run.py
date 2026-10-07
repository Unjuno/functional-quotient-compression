import argparse,csv,json,statistics,time
from pathlib import Path
import torch
from engine import METHODS,TASKS,D,O,world,pretrain_base,Learner,evaluate,mac_proxy

PRE_UPDATES,NEW_UPDATES,BATCH=600,300,64
FIELDS=['split','seed','condition','learning_rate','method','stage','tasks_seen','task0_mse','task1_mse','task2_mse','task3_mse','task4_mse','mean_seen_mse','forgetting_abs','shared_backbone_bytes','inference_payload_bytes','incremental_inference_bytes','inference_bytes_per_skill','resume_payload_bytes','resume_bytes_per_skill','train_examples_cumulative','optimizer_updates_cumulative','active_compute_proxy','wall_time_s','inference_examples_per_s']

def throughput(m,seed,task):
 g=torch.Generator().manual_seed(seed);x=torch.randn(64,D,generator=g)
 with torch.no_grad():
  for _ in range(5):m.forward(x,task)
  t=time.perf_counter()
  for _ in range(30):m.forward(x,task)
 return 64*30/(time.perf_counter()-t)

def run_world(seed,condition,lr,split):
 w=world(seed,condition);base_seed=seed+101;bw,bb,preopt,prewall,pre_examples=pretrain_base(w,base_seed,lr,PRE_UPDATES,BATCH)
 rows=[]
 for mi,method in enumerate(METHODS):
  learner=Learner(method,bw,bb,seed+mi*719,preopt if method=='sequential_finetune' else None,lr)
  base_bytes=learner.base_payload_bytes();base_resume_bytes=learner.resume_base_bytes()
  acquired={0:evaluate(learner,w,seed+50000)[0]}
  for task in range(1,TASKS):
   learner.acquire(task,w,lr,NEW_UPDATES,BATCH)
   mse=evaluate(learner,w,seed+50000)
   acquired[task]=mse[task]
   seen_errors=mse[:task+1]
   earlier=[mse[i]-acquired[i] for i in range(task)]
   forgetting=sum(earlier)/len(earlier) if earlier else 0.0
   inf=learner.inference_bytes();resume=learner.resume_bytes();skills=task;post_updates=0 if method=='hard_tie' else task*NEW_UPDATES;examples=pre_examples+post_updates*BATCH
   row={'split':split,'seed':seed,'condition':condition,'learning_rate':f'{lr:.3f}','method':method,'stage':task,'tasks_seen':task+1,
        'mean_seen_mse':f'{statistics.mean(seen_errors):.10g}','forgetting_abs':f'{forgetting:.10g}','shared_backbone_bytes':base_bytes,'inference_payload_bytes':inf,'incremental_inference_bytes':inf-base_bytes,'inference_bytes_per_skill':f'{(inf-base_bytes)/skills:.6f}',
        'resume_payload_bytes':resume,'resume_bytes_per_skill':f'{max(0,resume-base_resume_bytes)/skills:.6f}','train_examples_cumulative':examples,'optimizer_updates_cumulative':PRE_UPDATES+post_updates,
        'active_compute_proxy':mac_proxy(method,examples),'wall_time_s':f'{prewall+learner.wall:.6f}','inference_examples_per_s':f'{throughput(learner,seed+task*17,task):.3f}'}
   for i in range(TASKS):row[f'task{i}_mse']=f'{mse[i]:.10g}' if i<=task else ''
   rows.append(row)
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['development','fresh'],required=True);p.add_argument('--seeds',nargs='+',type=int,required=True);p.add_argument('--lr',type=float,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
 rows=[]
 for seed in a.seeds:
  for cond in ('aligned','independent'):
   got=run_world(seed,cond,a.lr,a.split);rows.extend(got)
   for r in got:print(a.split,seed,cond,r['method'],'stage',r['stage'],'mean',r['mean_seen_mse'],'forget',r['forgetting_abs'],'inferB/skill',r['inference_bytes_per_skill'],flush=True)
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n');w.writeheader();w.writerows(rows)
 print(json.dumps({'split':a.split,'seeds':a.seeds,'learning_rate':a.lr,'rows':len(rows)}))
if __name__=='__main__':main()
