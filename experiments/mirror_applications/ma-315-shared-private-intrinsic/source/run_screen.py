from __future__ import annotations
import csv,json
from pathlib import Path
from engine import FRACTIONS,run_world

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts';OUT.mkdir(exist_ok=True)
DEV=(31501,31502);FRESH=(31511,31512,31513)

def run(seeds,name):
    rows=[];events=[]
    for seed in seeds:
        for fraction in FRACTIONS:
            r,e=run_world(seed,fraction,OUT);rows.extend(r);events.extend(e)
    with (ROOT/f'{name}_RESULTS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (ROOT/f'{name}_EVENTS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(events[0]));w.writeheader();w.writerows(events)
    return rows

def gate(rows,seeds):
    verdict=[]
    for seed in seeds:
        for frac in (0.0,0.25):
            w=[r for r in rows if r['seed']==seed and r['private_fraction']==frac]
            m=next(r for r in w if r['method']=='mirror_sparse')
            c=next(r for r in w if r['method']=='adaptive_direct')
            checks={'quality':m['normalized_mse']<=1e-4,
                    'nearest_control_quality':m['normalized_mse']<=c['normalized_mse']+1e-5,
                    'bytes':m['serialized_bytes']<=.90*c['serialized_bytes']}
            verdict.append({'seed':seed,'private_fraction':frac,'checks':checks,'all_pass':all(checks.values())})
    return verdict

if __name__=='__main__':
    dev=run(DEV,'DEV');checks=gate(dev,DEV)
    (ROOT/'source/dev_gate.json').write_text(json.dumps(checks,indent=2)+'\n')
    if all(x['all_pass'] for x in checks):
        fresh=run(FRESH,'FRESH');v=gate(fresh,FRESH)
        (ROOT/'source/fresh_gate.json').write_text(json.dumps(v,indent=2)+'\n')
