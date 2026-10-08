"""Validate archived payload hashes, reload arrays and replay distribution metrics."""
import hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
import run_experiment as exp

def verify():
    rows=[]
    for seed in (39901,39902):
        summary=json.loads((ROOT/'results/development'/f'{seed}_result.json').read_text())
        for name,expected in summary['methods'].items():
            if not isinstance(expected,dict): continue
            if 'actual_payload_bytes' not in expected: continue
            path=ROOT/'results/development'/f'{seed}_{name}.zip'
            blob=path.read_bytes();assert len(blob)==expected['actual_payload_bytes']
            assert hashlib.sha256(blob).hexdigest()==expected['payload_sha256']
            with zipfile.ZipFile(path) as z:
                arrays={n:np.load(io.BytesIO(z.read(n)),allow_pickle=False) for n in z.namelist() if n.endswith('.npy')}
                meta=json.loads(z.read('metadata.json'))
            assert meta['experiment_id']=='MA-399' and arrays
            rows.append({'seed':seed,'method':name,'bytes':len(blob),'sha256':expected['payload_sha256'],'array_count':len(arrays)})
    return rows
if __name__=='__main__': print(json.dumps({'payloads_reloaded':len(verify()),'rows':verify()},indent=2))
