import argparse,csv,json,os,torch
torch.set_num_threads(1)
from engine import METHODS,ROLES,world,make_batch,teacher_logits,Student,evaluate,mac_proxy,benchmark_inference

def train_world(seed,condition,lr,updates):
 w=world(seed,condition);students={m:Student(m,w,seed+300000,lr) for m in METHODS}
 for s in students.values():s.set_teacher_reference(w)
 rows=[]
 for role in range(ROLES):
  x,_=make_batch(w,seed+100000+role,128,True);target=teacher_logits(x,w,role)
  for method,s in students.items():
   s.acquire(role,x,target,lr,updates);metrics=evaluate(s,w,seed+200000+role*101,role)
   speed=benchmark_inference(s,seed+500000+role) if role==ROLES-1 else {'inference_batch_size':0,'inference_repeats':0,'inference_wall_time_s':0.,'inference_examples_per_s':0.}
   for er,metric in enumerate(metrics):
    payload=s.serialize();rows.append({'condition':condition,'world_or_seed':seed,'stage_role':role,'evaluated_role':er,'method':method,'learning_rate':lr,'serialized_bytes':len(payload),'incremental_bytes':len(payload)-s.base_bytes(),'resume_bytes':s.resume_bytes(),'train_examples':s.examples.get(role,0),'optimizer_updates':s.updates.get(role,0),'active_compute_proxy':mac_proxy(method,s.examples.get(role,0),condition),'wall_time_s':s.wall.get(role,0.),'teacher_nll':metric['nll'],'teacher_kl':metric['kl'],'teacher_top1_agreement':metric['agreement'],'ece':metric['ece'],**speed})
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--seeds',required=True);p.add_argument('--conditions',default='aligned,independent');p.add_argument('--lrs',default='0.003,0.01');p.add_argument('--updates',type=int,default=300);p.add_argument('--out',required=True);a=p.parse_args();rows=[]
 for c in a.conditions.split(','):
  for seed in map(int,a.seeds.split(',')):
   for lr in map(float,a.lrs.split(',')):
    print(f'condition={c} seed={seed} lr={lr}',flush=True);rows.extend(train_world(seed,c,lr,a.updates))
 os.makedirs(os.path.dirname(a.out),exist_ok=True)
 with open(a.out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({'rows':len(rows),'out':a.out},sort_keys=True))
if __name__=='__main__':main()
