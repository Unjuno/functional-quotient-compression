"""Independent long-budget L1b frozen suite, using unmodified L1 kernels.

This changes only the sample worlds and update budget; training/data/model
implementations remain in the SHA-pinned source/l1.py.
"""
import argparse
import csv
import json
from pathlib import Path
import torch
from l1 import MODELS, TASKS, QueryNet, proxy_macs, run_one

WORLDS = (618211, 618212, 618213)
SEEDS = (0, 1)
BATCH = 32
UPDATES = 800
LR = 0.003
EVAL_N = 384


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',required=True,choices=('fixed','matched','hidden'))
    p.add_argument('--out',required=True)
    a=p.parse_args()
    out=Path(a.out)
    if out.exists():
        raise FileExistsError(f'Never overwrite evidence: {out}')
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    if a.mode=='fixed':
        tasks=('copy','retrieve','ordered')
        models=('native','tied','gain4','rank1_4','mirror4','independent4')
    elif a.mode=='matched':
        tasks=('copy','retrieve','ordered')
        models=('native','tied','gain4','rank1_4')
    else:
        tasks=('hidden',)
        models=('native','mirror4')
    rows=[]
    for task in tasks:
        assert task in TASKS
        for model in models:
            assert model in MODELS
            updates=UPDATES if a.mode!='matched' else max(1,int(UPDATES*proxy_macs(QueryNet('mirror4'))/proxy_macs(QueryNet(model))))
            for world in WORLDS:
                for seed in SEEDS:
                    row=run_one(task,model,world,seed,updates,BATCH,LR,EVAL_N)
                    row['budget_mode']=a.mode
                    rows.append(row)
                    print(f'{task} {model} {world} {seed} nll={row["nll"]:.4f} acc={row["accuracy"]:.3f} '+
                          f'updates={updates} time={row["train_seconds"]:.2f}s',flush=True)
                    with out.open('w',newline='') as f:
                        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
                        writer.writeheader();writer.writerows(rows)
    out.with_suffix('.meta.json').write_text(json.dumps({'stage':'L1b','mode':a.mode,
        'worlds':WORLDS,'seeds':SEEDS,'updates_reference':UPDATES,'batch':BATCH,'lr':LR,
        'eval_n':EVAL_N,'rows':len(rows),'cpu_threads':torch.get_num_threads(),
        'torch':torch.__version__},indent=2))

if __name__=='__main__':
    main()
