import argparse,csv
from engine import run

def main():
 p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',required=True);p.add_argument('--conditions',nargs='+',choices=['aligned','independent'],default=['aligned','independent']);p.add_argument('--output',required=True);p.add_argument('--updates',type=int,default=500);p.add_argument('--lr',type=float,default=.02);a=p.parse_args()
 cols=['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note']
 with open(a.output,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
  for c in a.conditions:
   for seed in a.seeds:
    for r in run(seed,c,a.updates,a.lr):
     w.writerow(dict(condition=c,world_or_seed=seed,method=r.method,serialized_bytes=r.bytes,train_tokens_or_examples=r.examples,optimizer_updates=r.updates,active_compute_proxy=r.macs,wall_time_s=f'{r.train_wall:.6f}',primary_metric='branch_verification_nll',primary_value=f'{r.nll:.10g}',secondary_metric='exact_path_accuracy',secondary_value=f'{r.path_acc:.10g}',status_note=f'inference_wall_s={r.infer_wall:.6f};branches_per_s={r.branches_s:.4f}'))
     print(c,seed,r.method,f'nll={r.nll:.5g}',f'path={r.path_acc:.3f}',f'bytes={r.bytes}',flush=True)
if __name__=='__main__':main()
