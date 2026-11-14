# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Independent A8 verification, exact witness and strict-no-shear sensitivity."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np
import a8

BASE=Path(__file__).resolve().parent

def check_run(root:Path)->dict:
    lock=json.loads((root/'train.lock.json').read_text())
    alock=json.loads((root/'audit.lock.json').read_text())
    assert hashlib.sha256((root/'train.lock.json').read_bytes()).hexdigest()==alock['train_lock_sha256']
    for name,digest in alock['file_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
    for name,meta in lock['files'].items():
        file=root/'models'/name
        assert file.stat().st_size==meta['bytes']
        assert hashlib.sha256(file.read_bytes()).hexdigest()==meta['sha256']
    return {'weights':lock['n'],'audit_scores':len(pd.read_csv(root/'audit.csv')),'hashes_verified':True}


def main()->None:
    main=BASE/'results';extended=BASE/'results_extended'
    x=check_run(main);y=check_run(extended)
    frontier=pd.read_csv(main/'frontier.csv')
    rows=[];witness=[]
    for (w,f,c),g in frontier.groupby(['world','family','cap']):
        m=g[g.kind=='mirror'].iloc[0]
        controls=g[~g.kind.isin(['mirror','pre'])].sort_values(['val_mse','kind'])
        b=controls.iloc[0]
        rows.append(dict(world=w,family=f,cap=c,selected_strict_control=b.kind,
                         mirror_iid_mse=m.iid_mse,control_iid_mse=b.iid_mse,
                         mirror_ood_mse=m.ood_mse,control_ood_mse=b.ood_mse,
                         mirror_joint_better=(m.iid_mse<b.iid_mse and m.ood_mse<b.ood_mse),
                         mirror_pass=bool(m.quality_pass),strict_control_pass=bool(b.quality_pass)))
    pd.DataFrame(rows).sort_values(['family','world','cap']).to_csv(main/'strict_controls.csv',index=False)
    for w in range(810,816):
        for f in ('hinge','ramp'):
            env=a8.make_world(w,f)
            for kind in ('mirror','pre'):
                model=a8.exact_construction(kind,env)
                for split in ('train','dev','iid','ood'):
                    r=a8.quality(model,env[split])
                    witness.append(dict(world=w,family=f,kind=kind,split=split,
                                        bytes=len(a8.serialize(model)),nrmse=r['nrmse'],
                                        max_abs=r['max_abs']))
    df=pd.DataFrame(witness)
    assert df.max_abs.max()<1e-6
    df.to_csv(main/'exact_witness.csv',index=False)
    result={'main':x,'extended':y,'witness_models':len(df.groupby(['world','family','kind'])),
            'witness_scores':len(df),'witness_max_abs':float(df.max_abs.max()),
            'strict_control_joint_wins':pd.DataFrame(rows).groupby(['family','cap']).mirror_joint_better.sum().to_dict(),
            'details':'Conditional shift/FiLM can also represent these tasks; exact witness is privileged teacher-derived, not optimization.'}
    def safe(v):
        if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
        if isinstance(v,np.integer):return int(v)
        return v
    (main/'verification.json').write_text(json.dumps(safe(result),indent=2)+'\n')
    print('A8_VERIFY_PASS',x,y,'witness_scores',len(df),'max',df.max_abs.max())

if __name__=='__main__': main()
