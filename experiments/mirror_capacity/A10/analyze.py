# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Post-lock descriptive cross-world comparisons, no model selection using test data."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
from scipy.stats import t


def summarize(root:Path)->dict:
    root=Path(root)
    ds={name:json.loads((root/name/'summary.json').read_text()) for name in ('main','a10b_short','a10b_long')}
    out=dict()
    for cap in (512,1024):
        rows=ds['main']['worlds']
        def get(method):return {r['world']:r for r in rows if r['cap']==cap and r['method']==method}
        m,f=get('mirror'),get('film')
        assert set(m)==set(f)==set(range(900,906))
        values=np.array([m[w]['new_iid_nrmse']/f[w]['new_iid_nrmse'] for w in sorted(m)])
        logs=np.log(values); mean=float(logs.mean()); u=float(logs.std(ddof=1)/math.sqrt(len(values)))
        k=float(t.ppf(.975,df=len(values)-1))
        out[f'main_cap{cap}']=dict(n_worlds=6,mirror_to_film_iid_geomean=math.exp(mean),
            min_ratio=float(values.min()),max_ratio=float(values.max()),
            world_wins=int((values<1).sum()),log_ratio_u_c=u,t95_k=k,
            ratio_t95=[math.exp(mean-k*u),math.exp(mean+k*u)])
    for meth in ('mirror','shift','film'):
        short={r['world']:r for r in ds['a10b_short']['worlds'] if r['method']==meth}
        long={r['world']:r for r in ds['a10b_long']['worlds'] if r['method']==meth}
        assert set(short)==set(long)==set(range(906,910))
        out[f'a10b_{meth}']=dict(n_worlds=4,
            iid_short=float(np.mean([short[w]['new_iid_nrmse'] for w in sorted(short)])),
            iid_long=float(np.mean([long[w]['new_iid_nrmse'] for w in sorted(short)])),
            ood_short=float(np.mean([short[w]['new_ood_nrmse'] for w in sorted(short)])),
            ood_long=float(np.mean([long[w]['new_ood_nrmse'] for w in sorted(short)])),
            pass_short=sum(r['passes'] for r in short.values()),
            pass_long=sum(r['passes'] for r in long.values()),
            iid_worlds_improved=sum(long[w]['new_iid_nrmse'] < short[w]['new_iid_nrmse'] for w in short))
    (root/'contrasts.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    return out

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    print(json.dumps(summarize(p.parse_args().root),indent=2))
