"""Locked contrasts use world-level paired units, not pooled pseudo-replication."""
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from engine import ROOT,CFG

def world_summary(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['world'],r['family'],r['ntrain'],r['method'])].append(r)
    out=[]
    for (w,f,n,m),rs in sorted(groups.items()):
        losses=np.array([r['test_mse'] for r in rs])
        out.append(dict(world=w,family=f,ntrain=n,method=m,test_mean=float(losses.mean()),seed_variance=float(losses.var(ddof=1)),seed_std=float(losses.std(ddof=1)),
                        seed_min=float(losses.min()),seed_max=float(losses.max()),worst_regime=float(np.mean([r['worst_mse'] for r in rs])),
                        train_mean=float(np.mean([r['train_mse'] for r in rs])),generalization_gap=float(np.mean([r['test_mse']-r['train_mse'] for r in rs])),
                        best_test_mean=float(np.mean([r['best_test_mse'] for r in rs])),starved=int(sum(min(r['usage'],1-r['usage'])<.05 for r in rs)),
                        route_match=float(np.mean([r['route_match'] for r in rs])),expert_difference=float(np.mean([r['expert_difference'] for r in rs])),bytes=int(rs[0]['bytes'])))
    return out

def write_csv(path,rows):
    with open(path,'w') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

def main():
    rs=json.loads((ROOT/'FRESH_RECORDS.json').read_text());ws=world_summary(rs)
    write_csv(ROOT/'RESULTS_CORE.csv',rs);write_csv(ROOT/'WORLD_SUMMARY.csv',ws)
    groups=defaultdict(list)
    for r in ws:groups[(r['family'],r['ntrain'],r['method'])].append(r)
    sums=[]
    for (fam,n,m),data in sorted(groups.items()):
        row={'family':fam,'ntrain':n,'method':m,'worlds':len(data),'inits_per_world':len(CFG['fresh_inits']),'bytes':data[0]['bytes']}
        for key in ['test_mean','seed_variance','seed_std','worst_regime','train_mean','generalization_gap','best_test_mean','route_match','expert_difference']:
            a=np.array([r[key] for r in data]);row[key+'_median']=float(np.median(a));row[key+'_min']=float(a.min());row[key+'_max']=float(a.max())
        row['starved_runs']=sum(r['starved'] for r in data);sums.append(row)
    write_csv(ROOT/'SUMMARY.csv',sums)
    contrasts=[('capacity','mirror8_free','mirror1_free'),('capacity4','mirror4_free','mirror1_free'),
               ('stability_capacity','mirror8_free','mirror1_free'),('stability_warm','mirror8_warm','mirror8_free'),
               ('stability_equal','mirror8_equal','mirror8_free'),('specificity','mirror8_free','gain8_free'),
               ('warm_mean','mirror8_warm','mirror8_free'),('gain_warm_mean','gain8_warm','gain8_free')]
    decisions=[];paired=[]
    for fam in CFG['families']:
      for n in CFG['train_counts']:
        for test,method,ref in contrasts:
            pairs=[]
            for w in CFG['fresh_worlds']:
                a=next(r for r in ws if (r['family'],r['ntrain'],r['method'],r['world'])==(fam,n,method,w))
                b=next(r for r in ws if (r['family'],r['ntrain'],r['method'],r['world'])==(fam,n,ref,w))
                quality=a['test_mean']/max(b['test_mean'],1e-12);worst=a['worst_regime']/max(b['worst_regime'],1e-12);variance=a['seed_variance']/max(b['seed_variance'],1e-12)
                passed=(variance<=.7 and quality<=1.01 and worst<=1.01) if test.startswith('stability') else (quality<=.95 and worst<=1.01)
                row=dict(family=fam,ntrain=n,test=test,method=method,reference=ref,world=w,quality_ratio=quality,worst_ratio=worst,variance_ratio=variance,passed=bool(passed))
                paired.append(row);pairs.append(row)
            delta=np.array([1-r['quality_ratio'] for r in pairs]);u=float(delta.std(ddof=1)/np.sqrt(len(delta)))
            decisions.append(dict(family=fam,ntrain=n,test=test,method=method,reference=ref,worlds_passed=sum(r['passed'] for r in pairs),worlds=len(pairs),verdict='PASS' if sum(r['passed'] for r in pairs)>=4 else 'FAIL',
              median_quality_ratio=float(np.median([r['quality_ratio'] for r in pairs])),median_variance_ratio=float(np.median([r['variance_ratio'] for r in pairs])),mean_relative_reduction=float(delta.mean()),u_c=u,k=2.776445105,interval_lower=float(delta.mean()-2.776445105*u),interval_upper=float(delta.mean()+2.776445105*u)))
    write_csv(ROOT/'PAIRED_WORLD.csv',paired);(ROOT/'DECISIONS.json').write_text(json.dumps(decisions,indent=2))
    print(json.dumps(decisions,indent=2))
if __name__=='__main__':main()
