"""Independent restore/metric/hash audit for stage A/B, not a training script."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import pandas as pd
import torch
from torch.nn import functional as F
from experiment import split_world,evaluate
from direct_labels import compute_metrics
from model import Teachers,Student,reload_payload
ROOT=Path(__file__).parent

def sha(b):return hashlib.sha256(b).hexdigest()

def run():
    torch.set_num_threads(1)
    check={'protocol_locks':{},'results':{}}
    for nm in ['FRESH_LOCK.json','DIRECT_LOCK.json']:
        lock=json.loads((ROOT/nm).read_text())
        for path,digest in lock['source_sha256'].items():
            if sha((ROOT/path).read_bytes())!=digest:raise AssertionError('Source modified after lock:'+path)
        check['protocol_locks'][nm]='PASS'
    rows_a=pd.read_csv(ROOT/'results/FRESH_CORE.csv')
    rows_b=pd.read_csv(ROOT/'results/DIRECT_FRESH.csv')
    assert len(rows_a)==108 and len(rows_b)==108
    for label,rows,metricsfn in [('A',rows_a,evaluate),('B',rows_b,compute_metrics)]:
        group=rows.groupby(['world','init' if label=='A' else 'seed','method'])
        assert len(group)==36 and len(rows)==108
        maxerr=0;count=0;bytes_sum=0
        for (world,initialization,method),records in group:
            path=ROOT/'models'/((f'{world}_{initialization}_{method}.pt') if label=='A' else f'direct_{world}_{initialization}_{method}.pt')
            b=path.read_bytes();rec=records.iloc[-1]
            assert len(b)==int(rec['inference_bytes']),(path,len(b),rec['inference_bytes'])
            assert sha(b)==rec['model_sha256'],path
            m=reload_payload(b);m.eval()
            data=split_world(int(world))
            if label=='A':
                tb=ROOT/'models'/f'teacher_world{int(world)}.pt'
                state=torch.load(tb,weights_only=True,map_location='cpu')
                teacher=Teachers();teacher.load_state_dict(state['state']);teacher.eval()
                found=metricsfn(m,teacher,data['test'])
            else:found=metricsfn(m,data['test'])
            for key,v in found.items():
                k='test_'+key
                if k in rec:
                    err=abs(float(rec[k])-v)
                    assert err<5e-7,(label,path,k,err)
                    maxerr=max(maxerr,err)
            count+=1;bytes_sum+=len(b)
        check['results'][label]={'payloads_verified':count,'max_abs_test_metric_error':maxerr,'total_payload_bytes':bytes_sum}
    (ROOT/'results/AUDIT.json').write_text(json.dumps(check,indent=2)+'\n')
    print(json.dumps(check,indent=2))
if __name__=='__main__':run()
