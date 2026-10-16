"""Post-hoc strong reference controls; no retuning or new claim promotion."""
import json
from concurrent.futures import ProcessPoolExecutor
from engine import ROOT,CFG,fit
SPEC=json.loads((ROOT/'WIDEN_PROTOCOL.json').read_text())
def task(args):
    w,f,m=args;out=ROOT/'results'/'widen_references'/f'w{w}_{f}'
    lr=json.loads((ROOT/'FRESH_LOCK.json').read_text())['learning_rate']
    r=fit(w,f,512,m,SPEC['inits'],lr,out)
    return r['records']
if __name__=='__main__':
    jobs=[(w,f,m) for w in SPEC['worlds'] for f in SPEC['families'] for m in ['mirror8_free','gain8_free']];records=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,rs in enumerate(pool.map(task,jobs),1):records+=rs;print('reference',i,len(jobs),flush=True)
    (ROOT/'WIDEN_REFERENCE_RECORDS.json').write_text(json.dumps(records,indent=2));print('COMPLETE',len(records),flush=True)
