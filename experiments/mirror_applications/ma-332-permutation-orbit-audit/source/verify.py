import csv, hashlib
from pathlib import Path
import numpy as np
import run
ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((ROOT/'FRESH_RESULTS.csv').open()))
maxdiff=0.0
for row in rows:
    world=int(row['world']); method=row['method']
    x,w1,b1,w2,b2,p,scale,negscale=run.arrays(world)
    blob=run.payload(method,w1,b1,w2,b2,p,scale,negscale)
    meta,base,code=run.load_payload(blob)
    assert meta['method']==method
    assert [a.shape for a in base]==[w1.shape,b1.shape,w2.shape,b2.shape]
    assert hashlib.sha256(blob).hexdigest()==row['payload_sha256']
    assert len(blob)==int(row['serialized_bytes'])
    p2=code if method=='permutation_view' else p
    s2=code if method=='positive_scale_compensated' else scale
    ns2=code if method=='negative_scale_compensated' else negscale
    ref=run.evaluate('baseline',x,w1,b1,w2,b2,p,scale,negscale)
    actual=run.evaluate(method,x,*base,p2,s2,ns2)
    replay=run.nrmse(actual,ref)
    maxdiff=max(maxdiff,abs(replay-float(row['output_nrmse'])))
assert maxdiff <= 1e-12, maxdiff
print({'rows':len(rows),'max_metric_difference':maxdiff,'roundtrip_payload_hashes':'exact'})
