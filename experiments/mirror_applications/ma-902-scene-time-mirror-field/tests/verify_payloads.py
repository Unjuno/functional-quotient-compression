import csv
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((ROOT/'source/results/RESULTS_CORE.csv').open()))
assert len(rows)==20
assert {int(r['seed']) for r in rows if r['condition'].startswith('fresh:')}=={90221,90223,90229}
for seed in (90221,90223,90229):
    r=[x for x in rows if x['condition']==f'fresh:mirror_multiplicative' and int(x['seed'])==seed][0]
    f=ROOT/'source/results/payloads'/f'fresh_{seed}_mirror_multiplicative.npz'
    with np.load(f,allow_pickle=False) as z:
        h,scene,t=z['H'],z['scene'],z['time']
        # Deterministic deployed decode for each logical scene-time pair.
        decoded=np.einsum('sr,tr,rx->stx',scene,t,h)
        assert np.isfinite(decoded).all()
    assert int(r['actual_payload_bytes'])==f.stat().st_size
manifest=json.loads((ROOT/'source/results/split_manifest.json').read_text())
assert all(d['observed_cells']==32 and d['heldout_cells']==16 for d in manifest)
print('payloads/replay/split integrity: PASS')
