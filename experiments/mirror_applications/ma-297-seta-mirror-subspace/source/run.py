import argparse, csv
from pathlib import Path
from engine import run

ROOT=Path(__file__).resolve().parents[1]
FIELDS=['split','seed','condition','learning_rate','method','tasks_seen','mean_seen_mse','forgetting_abs','shared_basis_bytes','inference_payload_bytes','incremental_inference_bytes','train_examples_cumulative','optimizer_updates_cumulative','active_compute_proxy','wall_time_s','inference_examples_per_s','payload_sha256','retention_task0_mse','reconstruction_max_abs_diff']

def main():
    p=argparse.ArgumentParser(); p.add_argument('--split',required=True); p.add_argument('--seeds',nargs='+',type=int,required=True); p.add_argument('--support-pairs',type=int,required=True); a=p.parse_args()
    out=ROOT/f'{a.split.upper()}_RAW.csv'
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader()
        for seed in a.seeds:
            for row in run(seed,a.split,0.0,a.support_pairs): w.writerow(row)
    print(out)
if __name__=='__main__': main()
