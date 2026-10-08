import argparse,csv
from pathlib import Path
from engine import run
ROOT=Path(__file__).resolve().parents[1]
FIELDS=['split','seed','condition','method','rank','depth','layer_output_mse','composed_output_mse','serialized_bytes','optimizer_updates','active_compute_proxy','operator_prepare_s','wall_time_s','inference_examples_per_s','payload_sha256','reconstruction_max_abs_diff','runtime_path_max_abs_diff']
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',required=True);p.add_argument('--seeds',nargs='+',type=int,required=True);p.add_argument('--rank',type=int,required=True);a=p.parse_args();out=ROOT/f'{a.split.upper()}_RAW.csv'
 with out.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader()
  for seed in a.seeds:
   for row in run(seed,a.split,a.rank):w.writerow(row)
 print(out)
if __name__=='__main__':main()
