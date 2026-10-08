from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import numpy as np
from engine import FRACTIONS,METHODS,make_world,make_state,save_payload,load_payload,decode,_norm_mse,T,NT

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';maxdiff=0.0;checked=0;event_rows_checked=0;expected_events={}
for name in ('DEV','FRESH'):
    result_file=ROOT/f'{name}_RESULTS.csv';event_file=ROOT/f'{name}_EVENTS.csv'
    if not result_file.exists():continue
    with result_file.open(newline='') as f:rows=list(csv.DictReader(f))
    for row in rows:
        seed=int(row['seed']);fraction=float(row['private_fraction']);method=row['method']
        theta0,basis,private,angles,coeff,targets,xtr,xval,xte,ytr,yval,yte=make_world(seed,fraction)
        state,ops=make_state(method,theta0,basis,xtr,ytr,xval,yval)
        path=OUT/f'{seed}_p{int(fraction*100):03d}_{method}.npz'
        nbytes=save_payload(path,state);assert nbytes==int(row['serialized_bytes'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['payload_sha256']
        loaded=load_payload(path)
        metrics=[_norm_mse(xte[t],yte[t],decode(method,loaded,t))[1] for t in range(T)]
        diff=abs(float(np.mean(metrics))-float(row['normalized_mse']));maxdiff=max(maxdiff,diff);assert diff<1e-12
        assert int(row['fit_compute_proxy'])==ops
        for t,val in enumerate(metrics):
            if method=='adaptive_direct':dim=int(loaded['dims'][t])
            elif method=='mirror_sparse':dim=2+int(loaded['dims'][t])
            elif method=='fixed_said8':dim=8
            else:dim=0
            expected_events[(seed,fraction,method,t)]=(int(private[t]),dim,val)
        checked+=1
    with event_file.open(newline='') as f:events=list(csv.DictReader(f))
    assert len(events)==len(rows)*T
    for e in events:
        key=(int(e['seed']),float(e['private_fraction']),e['method'],int(e['task']))
        truth=expected_events[key]
        assert int(e['is_private'])==truth[0] and int(e['selected_intrinsic_dimension'])==truth[1]
        de=abs(float(e['test_normalized_mse'])-truth[2]);maxdiff=max(maxdiff,de);assert de<1e-12
        event_rows_checked+=1
result={'result_rows_replayed':checked,'payload_hashes_checked':checked,'max_metric_difference':maxdiff,'event_rows_checked':event_rows_checked,'serialization_byte_exact':True}
(ROOT/'source/replay_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
