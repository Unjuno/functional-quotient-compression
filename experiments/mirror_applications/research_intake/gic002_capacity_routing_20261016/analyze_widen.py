"""Independent audit and paired summaries for the locked widening probe."""
import json
from pathlib import Path
import numpy as np
from engine import ROOT,sha,world
from analyze import world_summary,write_csv
from audit import numpy_predict

def main():
    rows=json.loads((ROOT/'WIDEN_RECORDS.json').read_text());ws=world_summary(rows)
    write_csv(ROOT/'WIDEN_RESULTS_CORE.csv',rows);write_csv(ROOT/'WIDEN_WORLD.csv',ws)
    specs=json.loads((ROOT/'WIDEN_PROTOCOL.json').read_text());dec=[]
    for fam in specs['families']:
        for method in specs['arms'][1:]:
            qs=[];vs=[];passed=0
            for w in specs['worlds']:
                a=next(r for r in ws if (r['world'],r['family'],r['method'])==(w,fam,method));b=next(r for r in ws if (r['world'],r['family'],r['method'])==(w,fam,'continue1'))
                qs.append(a['test_mean']/b['test_mean']);vs.append(a['seed_variance']/max(b['seed_variance'],1e-12))
                passed+=a['test_mean']<=.95*b['test_mean'] and a['worst_regime']<=1.01*b['worst_regime']
            dec.append(dict(family=fam,method=method,worlds_passed=int(passed),worlds=5,verdict='PASS' if passed>=4 else 'FAIL',quality_ratio_median=float(np.median(qs)),variance_ratio_median=float(np.median(vs)),quality_ratios=qs,variance_ratios=vs))
    (ROOT/'WIDEN_DECISIONS.json').write_text(json.dumps(dec,indent=2))
    payloads=0;maxerr=0.;insertion=0.;cache={}
    for row in rows:
        path=ROOT/row['file'];assert sha(path)==row['sha256'];assert path.stat().st_size==row['bytes']
        key=(row['world'],row['family'])
        if key not in cache:cache[key]=world(*key)
        d=cache[key];pred=numpy_predict(path,d['test_x'],d['test_c']);loss=float(np.mean((pred-d['test_clean'])**2));maxerr=max(maxerr,abs(loss-row['test_mse']))
        assert np.isclose(loss,row['test_mse'],atol=5e-6,rtol=2e-5)
        if row['method']=='expand8zero':insertion=max(insertion,row['insertion_error']);assert row['insertion_error']<=2e-6
        payloads+=1
    prefix_count=0
    for f in (ROOT/'results'/'widen').glob('*/prefix/mirror1_free.json'):
        for row in json.loads(f.read_text())['records']:
            d=cache[(row['world'],row['family'])]
            for ep in ('','best_'):
                path=ROOT/row[ep+'file'];assert sha(path)==row[ep+'sha256'] and path.stat().st_size==row['bytes']
                pred=numpy_predict(path,d['test_x'],d['test_c']);err=abs(float(np.mean((pred-d['test_clean'])**2))-row[ep+'test_mse']);maxerr=max(maxerr,err);assert err<5e-6;payloads+=1
            prefix_count+=1
    lock=json.loads((ROOT/'WIDEN_LOCK.json').read_text());matched=all(sha(ROOT/p)==h for p,h in lock['files'].items());assert matched
    report={'continuations':len(rows),'prefix_models':prefix_count,'payloads':payloads,'max_numpy_metric_error':maxerr,'zero_extension_max_function_change':insertion,'source_lock_match':matched}
    (ROOT/'WIDEN_VERIFICATION.json').write_text(json.dumps(report,indent=2));print(json.dumps(dec,indent=2));print(report)
if __name__=='__main__':main()
