"""Build raw-count tables and paired decisions from stored result JSON files."""
from pathlib import Path
import csv,json
import numpy as np

def main():
    rows=[]
    def add(stage,world,method,value,seed=None):
        ev=value['metrics']
        row=dict(stage=stage,world=world,init=seed,method=method,bytes=value['bytes'],
                 pair_correct=ev['audit_correct'],pair_total=ev['audit_n'],pair_acc=ev['audit_acc'],
                 atomic_correct=ev['atomic_correct'],atomic_total=ev['atomic_n'],exact_rules=ev['exact_rules'],
                 reverse_both_acc=ev['reverse_both_acc'])
        if 'replay_after' in value:row['two_call_replay_acc']=value['replay_after']['accuracy']
        rows.append(row)
    for file in sorted(Path('results').glob('fresh*/result.json')):
        r=json.loads(file.read_text())
        for v in r['results']:add('primary1600',r['world'],v['kind'],v,r['init'])
    for file in sorted(Path('results').glob('fresh*/long_result.json')):
        for v in json.loads(file.read_text()):add('long6400',v['world'],v['kind'],v)
    for file in sorted(Path('results').glob('paironly*/result.json')):
        r=json.loads(file.read_text())
        for v in r['results']:add('paironly1600',r['world'],v['kind'],v,r['init'])
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with open('ALL_RESULTS.csv','w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    groups=[]
    for stage,method in sorted(set((r['stage'],r['method']) for r in rows)):
        x=[r for r in rows if r['stage']==stage and r['method']==method]
        g=dict(stage=stage,method=method,n=len(x),bytes=x[0]['bytes'])
        for key in ['pair_acc','exact_rules','atomic_correct','reverse_both_acc']:
            a=[v[key] for v in x];g[key]=dict(median=float(np.median(a)),min=float(min(a)),max=float(max(a)))
        groups.append(g)
    by={(r['stage'],r['world'],r['method']):r for r in rows};comparison={}
    worlds=[66001,66002,66003]
    for a,b in [('hybrid','moe'),('hybrid','lora'),('selected','hybrid'),('selected','random'),('selected','small')]:
        d=[100*(by['primary1600',w,a]['pair_acc']-by['primary1600',w,b]['pair_acc']) for w in worlds]
        comparison[a+'_minus_'+b]=dict(deltas_pp=d,median_pp=float(np.median(d)),wins=sum(t>0 for t in d))
    summary=dict(groups=groups,comparisons=comparison,prune_fraction_total_bytes=1-by['primary1600',66001,'selected']['bytes']/by['primary1600',66001,'hybrid']['bytes'])
    Path('SUMMARY.json').write_text(json.dumps(summary,indent=2))
    gates=dict(primary_hybrid_2pp=all(comparison['hybrid_minus_'+b]['deltas_pp'][i]>=2 for b in ['moe','lora'] for i in range(3)),
               numerical_prune_noninferiority=all(d>=-1 for d in comparison['selected_minus_hybrid']['deltas_pp']),
               selected_beats_random_all=all(d>0 for d in comparison['selected_minus_random']['deltas_pp']),
               selected_beats_small_all=all(d>0 for d in comparison['selected_minus_small']['deltas_pp']),
               caveat='Development readiness failed; numerical noninferiority at low pair accuracy is not useful-quality compression.')
    Path('DECISIONS.json').write_text(json.dumps(gates,indent=2))
    print('Summary rows:',len(rows),'gates:',gates)
if __name__=='__main__':main()
