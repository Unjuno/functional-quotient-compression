"""MA-397 payload and collision metric replay verifier."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from model import from_payload
from run import make_world,score

def verify(path:Path)->dict[str,object]:
    d=json.loads(path.read_text());w=make_world(int(d['seed']));checks=[]
    assert d['unique_addresses']==1024 and d['address_collision_count']==1024
    assert d['address_occupancy_min']==d['address_occupancy_max']==2
    for r in d['results']:
        p=path.parent/r['payload_path'];raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest()
        assert len(raw)==r['serialized_bytes'] and h==r['payload_sha256']
        with np.load(p,allow_pickle=False) as z:bank=from_payload({k:z[k] for k in z.files})
        assert bank.method==r['method'];m=score(bank,w)
        for k,v in m.items():assert abs(v-r[k])<=1e-7,(r['method'],k,v,r[k])
        checks.append({'method':r['method'],'bytes':len(raw),'sha256':h,'metrics':m})
    return {'experiment_id':'MA-397','seed':d['seed'],'condition':d['condition'],'unique_addresses':d['unique_addresses'],
        'collision_count':d['address_collision_count'],'occupancy':2,'payloads_checked':len(checks),'all_exact':True,'payloads':checks}

def main()->None:
    p=argparse.ArgumentParser();p.add_argument('result',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    s=json.dumps(verify(a.result),indent=2)+'\n'
    if a.output:a.output.write_text(s)
    print(s,end='')
if __name__=='__main__':main()
