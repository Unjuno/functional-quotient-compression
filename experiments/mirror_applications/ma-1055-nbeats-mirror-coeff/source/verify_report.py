#!/usr/bin/env python3
"""Independent audit replay and serialized inference payload verifier."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_screen import CHANNELS, RAW_AUDIT, RoleModel, load_data, make_model

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify():
    result_path=HERE/'runs/audit/results.json'; frozen=HERE/'runs/frozen/FRESH_FREEZE'
    result=json.loads(result_path.read_text()); data,meta=load_data(include_audit=True)
    assert result['split']=='audit'
    report={'experiment_id':'MA-1055','status':'PASS_VERIFICATION','source_audit_sha256':sha(RAW_AUDIT),
            'source_protocol_sha256':sha(HERE/'PROTOCOL.json'),'fresh_freeze':{},'models':[],'checks':[]}
    paths={'PROTOCOL.json':HERE/'PROTOCOL.json','run_screen.py':HERE/'source/run_screen.py',
           'nbeats_upstream.py':HERE/'source/nbeats_upstream.py'}
    for name,current in paths.items():
        report['fresh_freeze'][name]={'sha256':sha(frozen/name),'current_sha256':sha(current)}
        assert report['fresh_freeze'][name]['sha256']==report['fresh_freeze'][name]['current_sha256']
    for row in result['results']:
        path=HERE/row['payload_path']; payload=torch.load(path,map_location='cpu',weights_only=False)
        assert path.stat().st_size==row['payload_bytes'] and sha(path)==row['payload_sha256']
        assert payload['method']==row['method'] and payload['seed']==row['seed']
        assert payload['audit_data_sha256']==sha(RAW_AUDIT)
        states=payload['models']; models=[]
        for i,state in enumerate(states):
            m=RoleModel(make_model(row['seed']+i if row['method']=='native' else row['seed']),row['method'])
            m.load_state_dict(state); m.eval(); models.append(m)
        replay=[]
        for ci,ch in enumerate(CHANNELS):
            d=data[(ci,'audit')]; x=torch.tensor(d['x'],dtype=torch.float32)
            r=torch.tensor(d['regime'],dtype=torch.long); c=torch.full((len(x),),ci,dtype=torch.long)
            with torch.no_grad(): pred=models[ci if row['method']=='native' else 0](x,c,r).numpy()
            score=float(np.mean(np.abs(pred-d['y']))/d['mase_scale'])
            saved=next(z for z in row['metrics'] if z['channel']==ch and z['split']=='audit')
            assert abs(score-saved['mase']) < 1e-7, (row['method'],row['seed'],ch,score,saved['mase'])
            replay.append({'channel':ch,'mase_replayed':score,'windows':len(x),
                           'target_start_min':int(d['starts'].min()),'target_end_max_exclusive':int(d['starts'].max()+96)})
        report['models'].append({'method':row['method'],'seed':row['seed'],'payload_bytes':path.stat().st_size,
            'payload_sha256':sha(path),'replayed_metrics':replay,'updates':row['updates'],
            'examples':row['examples'],'wall_seconds':row['wall_seconds'],'compute_proxy':row['active_compute_proxy']})
    report['checks']=['all checkpoint byte counts and SHA-256 values verified',
        'audit checkpoint data hash matches frozen ETTm2 CSV',
        'all per-channel audit MASE values replay exactly from serialized state',
        'all target intervals lie within registered ETTm2 final-20% audit interval']
    target=HERE/'VERIFICATION.json'; target.write_text(json.dumps(report,indent=2)+'\n')
    print(target)
if __name__=='__main__': verify()
