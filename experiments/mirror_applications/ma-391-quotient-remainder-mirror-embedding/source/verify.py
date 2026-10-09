"""Replay MA-391 serialized payloads and reported metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from run import make_world, score
from model import from_payload


def verify(result_path: Path) -> dict[str, object]:
    result=json.loads(result_path.read_text())
    world=make_world(int(result['seed']))
    assert result['unique_address_pairs']==1024
    checked=[]
    for row in result['results']:
        path=result_path.parent/row['payload_path']; raw=path.read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        assert len(raw)==row['serialized_bytes']
        assert digest==row['payload_sha256']
        with np.load(path,allow_pickle=False) as z: payload={key:z[key] for key in z.files}
        bank=from_payload(payload)
        assert bank.method==row['method']
        metrics=score(bank,world)
        for key,value in metrics.items(): assert abs(value-row[key])<=1e-7,(row['method'],key,value,row[key])
        checked.append({'method':row['method'],'bytes':len(raw),'sha256':digest,'metrics':metrics})
    return {'experiment_id':'MA-391','seed':result['seed'],'condition':result['condition'],
            'unique_address_pairs':result['unique_address_pairs'],'payloads_checked':len(checked),
            'all_exact':True,'payloads':checked}


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('result',type=Path); p.add_argument('--output',type=Path)
    a=p.parse_args(); report=verify(a.result); encoded=json.dumps(report,indent=2)+'\n'
    if a.output:a.output.write_text(encoded)
    print(encoded,end='')


if __name__=='__main__':main()
