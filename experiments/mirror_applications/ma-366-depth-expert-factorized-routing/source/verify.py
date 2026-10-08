import csv, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run

root=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((root/'artifacts/results.csv').open()))
assert len(rows)==20, len(rows)
assert {int(r['seed']) for r in rows}=={36601,36602,36611,36612,36613}
by={(int(r['seed']),r['method']):r for r in rows}
for seed in (36601,36602):
    for method in ('flat_paths','pa02_factorized','mirror_path','direct_coefficients'):
        r=by[seed,method]
        assert float(r['heldout_nMSE']) <= 1e-6
        assert int(r['distinct_paths']) == 16
    m=by[seed,'mirror_path']; d=by[seed,'direct_coefficients']
    assert m['payload_bytes']==d['payload_bytes']
    assert m['sha256'] != d['sha256'] # metadata names differ; decoded function is checked below
    assert m['route_MAC_proxy']==d['route_MAC_proxy']
    w=run.world(seed)
    for i,j in w[7]:
        assert np.array_equal(run.decode('mirror_path',run.states(w)['mirror_path'][0],i,j),
                              run.decode('direct_coefficients',run.states(w)['direct_coefficients'][0],i,j))
for seed in (36601,36602):
    assert int(by[seed,'pa02_factorized']['payload_bytes']) < int(by[seed,'mirror_path']['payload_bytes'])
assert json.loads((root/'PROTOCOL.json').read_text())['fresh']['deviation']
print('MA-366 verification passed: development exact reconstruction, direct-control parity, payloads; fresh rows disclosed and excluded due to protocol deviation.')
