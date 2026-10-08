import argparse,csv
from engine import run

def main():
 p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',required=True);p.add_argument('--conditions',nargs='+',choices=['aligned','independent'],default=['aligned','independent']);p.add_argument('--output',required=True);a=p.parse_args()
 cols=['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note']
 with open(a.output,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
  for c in a.conditions:
   for seed in a.seeds:
    for r in run(seed,c):
     w.writerow(dict(condition=c,world_or_seed=seed,method=r.method,serialized_bytes=r.payload_bytes,train_tokens_or_examples=512,optimizer_updates=0,active_compute_proxy=r.decode_macs,wall_time_s=f'{r.wall_s:.8f}',primary_metric='relative_frobenius_error',primary_value=f'{r.relative_fro:.10g}',secondary_metric='activation_output_mse',secondary_value=f'{r.activation_mse:.10g}',status_note='post_training_decode;20 repeated forward calls; no gradient updates'))
     print(c,seed,r.method,f'rel={r.relative_fro:.5g}',f'act={r.activation_mse:.5g}',f'bytes={r.payload_bytes}',flush=True)
if __name__=='__main__':main()
