from pathlib import Path
import hashlib,json,csv,torch
from experiment import split_world
from full_supervision import train
ROOT=Path(__file__).parent

def main():
    torch.set_num_threads(1)
    c=json.loads((ROOT/'LONG_FULL_PROTOCOL.json').read_text())
    lock=json.loads((ROOT/'LONG_FULL_LOCK.json').read_text())
    for f,h in lock['source_sha256'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,f
    rows=[]
    for world in c['worlds']:
        data=split_world(world)
        for s in c['initializations']:
            for method in c['model_methods']:
                got=train(world,s,method,data,c['lr'],c['updates'],c['checkpoints'],ROOT/'models')
                # rename to prevent confusion with previous runs
                old=ROOT/'models'/f'full_{world}_{s}_{method}.pt'
                new=ROOT/'models'/f'long_{world}_{s}_{method}.pt'
                old.rename(new)
                rows+=got
                print('LONG',world,s,method,round(got[-1]['test_all_BCE'],6),'sec',round(got[-1]['wall_s'],2),flush=True)
    with (ROOT/'results/LONG_FULL.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print('long models',len(rows)//len(c['checkpoints']))
if __name__=='__main__':main()
