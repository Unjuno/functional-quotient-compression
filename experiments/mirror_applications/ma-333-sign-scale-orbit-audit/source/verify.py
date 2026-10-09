import csv,hashlib
from pathlib import Path
import numpy as np
import run
ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((ROOT/'FRESH_RESULTS.csv').open()));maxdiff=0.
for r in rows:
 x,w1,b1,w2,b2,g,b,p,pos,sg=run.params(int(r['world']))
 blob=run.pack(r['activation'],r['method'],w1,b1,w2,b2,g,b,p,pos,sg)
 meta,arr,code=run.load(blob)
 assert len(blob)==int(r['serialized_bytes']) and hashlib.sha256(blob).hexdigest()==r['payload_sha256']
 if r['method']=='permutation_view': p=code
 if r['method']=='positive_scale_compensated': pos=code
 if r['method']=='sign_flip_compensated': sg=code
 if r['activation']=='layernorm_relu': w1,b1,w2,b2,g,b=arr
 else: w1,b1,w2,b2=arr
 ref=run.evaluate(r['activation'],'baseline',x,*run.params(int(r['world']))[1:5],*run.params(int(r['world']))[5:7],p,pos,sg)
 out=run.evaluate(r['activation'],r['method'],x,w1,b1,w2,b2,g,b,p,pos,sg)
 maxdiff=max(maxdiff,abs(run.nr(out,ref)-float(r['output_nrmse'])))
assert maxdiff<1e-12,maxdiff
print({'rows':len(rows),'max_metric_difference':maxdiff,'payload_hashes':'exact'})
