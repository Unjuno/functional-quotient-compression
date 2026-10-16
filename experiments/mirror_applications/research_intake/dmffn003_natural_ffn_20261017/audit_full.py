"""Audit of separate stage-C full supervision evidence."""
from __future__ import annotations
import json,hashlib
from pathlib import Path
import pandas as pd
import torch
from model import reload_payload
from experiment import split_world
from full_supervision import compute_metrics
ROOT=Path(__file__).parent

def run():
    torch.set_num_threads(1)
    lock=json.loads((ROOT/'FULL_LOCK.json').read_text())
    for fn,sha in lock['source_sha256'].items():
        assert hashlib.sha256((ROOT/fn).read_bytes()).hexdigest()==sha,fn
    df=pd.read_csv(ROOT/'results/FULL_FRESH.csv')
    assert len(df)==108
    count=0;maxerr=0.0
    for (world,seed,method),part in df.groupby(['world','seed','method']):
        x=ROOT/'models'/f'full_{world}_{seed}_{method}.pt'
        data=x.read_bytes();assert hashlib.sha256(data).hexdigest()==part.iloc[0]['model_sha256']
        assert len(data)==int(part.iloc[0]['inference_bytes'])
        test=split_world(int(world))['test']
        metrics=compute_metrics(reload_payload(data),test)
        row=part.iloc[-1]
        for k,v in metrics.items():
            err=abs(v-row['test_'+k]);assert err<1e-6
            maxerr=max(maxerr,float(err))
        count+=1
    assert count==36
    record={'frozen_source_hashes':'PASS','models_audited':count,'model_sha_and_bytes':'PASS','max_abs_metric_error':maxerr}
    (ROOT/'results/FULL_AUDIT.json').write_text(json.dumps(record,indent=2)+'\n');print(record)
if __name__=='__main__':run()
