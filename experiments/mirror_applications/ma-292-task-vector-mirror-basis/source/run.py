import argparse,csv
from pathlib import Path
from engine import run
ROOT=Path(__file__).resolve().parents[1]
FIELDS=['split','seed','condition','support_tasks','method','task_count','task_output_mse','max_task_output_mse','pair_arithmetic_mse','serialized_bytes','incremental_task_bytes','optimizer_updates','task_vectors_seen','active_compute_proxy','compression_wall_time_s','inference_examples_per_s','payload_sha256','max_delta_abs_error']
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',required=True);p.add_argument('--seeds',nargs='+',type=int,required=True);p.add_argument('--support-tasks',type=int,required=True);a=p.parse_args();out=ROOT/f'{a.split.upper()}_RAW.csv'
 with out.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader()
  for seed in a.seeds:
   for row in run(seed,a.split,a.support_tasks):w.writerow(row)
 print(out)
if __name__=='__main__':main()
