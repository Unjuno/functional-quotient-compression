"""Exact byte/hash/metric replay verifier for MA-393."""
from __future__ import annotations

import argparse,hashlib,json
from pathlib import Path
import numpy as np
from model import from_payload
from run import make_world,score


def verify(result_path:Path)->dict[str,object]:
    result=json.loads(result_path.read_text());world=make_world(int(result['seed']));checks=[]
    for row in result['results']:
        path=result_path.parent/row['payload_path'];raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        assert len(raw)==row['serialized_bytes'] and digest==row['payload_sha256']
        with np.load(path,allow_pickle=False) as z:payload={k:z[k] for k in z.files}
        bank=from_payload(payload);assert bank.method==row['method']
        metrics=score(bank,world)
        for key,value in metrics.items():assert abs(value-row[key])<=1e-7,(row['method'],key,value,row[key])
        checks.append({'method':row['method'],'bytes':len(raw),'sha256':digest,'metrics':metrics})
    return {'experiment_id':'MA-393','seed':result['seed'],'condition':result['condition'],
            'payloads_checked':len(checks),'all_exact':True,'checks':checks}


def main()->None:
    p=argparse.ArgumentParser();p.add_argument('result',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();text=json.dumps(verify(a.result),indent=2)+'\n'
    if a.output:a.output.write_text(text)
    print(text,end='')


if __name__=='__main__':main()
