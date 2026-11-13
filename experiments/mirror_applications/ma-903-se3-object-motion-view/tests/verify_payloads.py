import csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((ROOT/'source/results/RESULTS_CORE.csv').open()))
assert len(rows)==15
for seed in (90331,90333,90339):
    m=ROOT/'source/results/payloads'/f'fresh_{seed}_mirror_se3.npz'
    n=ROOT/'source/results/payloads'/f'fresh_{seed}_native_se3.npz'
    assert m.read_bytes()==n.read_bytes()
    with np.load(m,allow_pickle=False) as z:
        for M in z['actions']:
            R=M[:3,:3]
            assert np.max(np.abs(R@R.T-np.eye(3)))<2e-5
            assert np.linalg.det(R)>.999
    mr=next(r for r in rows if r['condition']=='fresh:mirror_se3' and int(r['seed'])==seed)
    nr=next(r for r in rows if r['condition']=='fresh:native_se3' and int(r['seed'])==seed)
    assert mr['actual_payload_bytes']==nr['actual_payload_bytes']
    assert 'max_error=0' in mr['quality'] and 'max_error=0' in nr['quality']
manifest=json.loads((ROOT/'source/results/split_manifest.json').read_text())
assert {d['seed'] for d in manifest if d['fresh']}=={90331,90333,90339}
print('SE3 validity, exact alias, payload sizes, and fresh split: PASS')
