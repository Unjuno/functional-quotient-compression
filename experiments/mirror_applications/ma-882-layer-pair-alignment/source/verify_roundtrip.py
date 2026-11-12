"""Replay saved MA-882 inference payloads on the frozen fresh test rows."""
import importlib.util, json, hashlib, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
EXP=importlib.util.spec_from_file_location('ma882',Path(__file__).with_name('experiment.py'))
m=importlib.util.module_from_spec(EXP); EXP.loader.exec_module(m)
raw_results=json.loads((ROOT/'fresh_results_pre_payload_roundtrip.json').read_text())
models,toks,_=m.load_assets(); replay_rows=[]
for row in raw_results['rows']:
    seed=row['seed']; variants={}; routes=None; hashes={}; bytes_by={}; decode_sec={}
    for name in ('full','pca','cp'):
        path=ROOT/'artifacts'/f'{name}_seed_{seed}.bin'; raw=path.read_bytes(); t0=time.perf_counter()
        decoded_name,bank,decoded_routes=m.decode_payload(raw); decode_sec[name]=time.perf_counter()-t0
        assert decoded_name==name
        variants[name]=bank
        if routes is None: routes=decoded_routes
        else: assert all(np.array_equal(a,b) for a,b in zip(routes,decoded_routes))
        hashes[name]=hashlib.sha256(raw).hexdigest(); bytes_by[name]=len(raw)
        assert bytes_by[name]==row['payload_bytes'][name]
        assert hashes[name]==row['payload_sha256'][name]
    tests=[m.prompt_chunks('test',pair,toks,seed,'test') for pair in m.PAIRS]
    ev=m.evaluate(variants,routes,tests,[],models)
    pair_diffs=[]
    for pi,pr in enumerate(row['pair_results']):
        sl=slice(pi*12,(pi+1)*12); fresh=ev['methods']
        pair_diffs.append({name:float(np.mean(fresh[name]['nll'][sl])-pr[f'{name}_nll']) for name in variants})
    maxdiff=max(abs(v) for d in pair_diffs for v in d.values())
    assert maxdiff<1e-3, (seed,pair_diffs)
    replay_rows.append({'seed':seed,'payload_bytes':bytes_by,'payload_sha256':hashes,'decode_seconds':decode_sec,'pair_mean_nll_diff':pair_diffs,'max_abs_mean_nll_diff':maxdiff,'native_identity_max_abs_nll_diff':ev['identity_max_abs_nll_diff']})
out={'experiment_id':'MA-882','replay_type':'same-fresh-prompt payload reconstruction diagnostic; not independent audit','rows':replay_rows}
(ROOT/'payload_roundtrip_replay.json').write_text(json.dumps(out,indent=2)+'\n')
for row in replay_rows: print(json.dumps(row,sort_keys=True))
