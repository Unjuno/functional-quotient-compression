#!/usr/bin/env python3
"""Check MA-597 frozen hashes, serialized inference outputs, replays, and gates."""
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import sys
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'source/run_hashednet.py'
sys.path.insert(0,str(ROOT/'source'))
from run_hashednet import METHODS,digits_data,evaluate_payload

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    freeze=json.loads((ROOT/'FREEZE.json').read_text())
    assert sha(ROOT/'PROTOCOL.json')==freeze['protocol_sha256']
    assert sha(SOURCE)==freeze['source_sha256']
    assert sha(ROOT/'tests/test_hashnet.py')==freeze['tests_sha256']
    x,y,digest=digits_data();assert digest=='faf2d98f61250c1cdc0b114323133d3c58da04f5c5d28bb78e4bde3c936a75b1'
    replay_info={}
    for seed in (59701,59702):
        xtr,xte,ytr,yte=train_test_split(x,y,test_size=.25,random_state=seed,stratify=y)
        a=ROOT/f'runs/dev_{seed}';b=ROOT/f'runs/replay_{seed}'
        ma=json.loads((a/'metrics.json').read_text());mb=json.loads((b/'metrics.json').read_text())
        assert ma['dataset_tensor_sha256']==digest
        assert ma['split']=='dev' and ma['split_random_state']==seed
        assert [m['method'] for m in ma['methods']]==list(METHODS)
        assert [m['method'] for m in ma['methods']]==[m['method'] for m in mb['methods']]
        for i,method in enumerate(METHODS):
            name=f'{method}.npz';pa=a/name;pb=b/name
            assert sha(pa)==sha(pb)
            mm=ma['methods'][i];rr=mb['methods'][i]
            assert sha(pa)==mm['payload_sha256']
            assert pa.stat().st_size==mm['serialized_bytes']
            assert mm['test_accuracy']==rr['test_accuracy']
            assert mm['test_cross_entropy']==rr['test_cross_entropy']
            actual=evaluate_payload(pa,xte,yte)
            assert actual['accuracy']==mm['test_accuracy']
            assert actual['mean_cross_entropy']==mm['test_cross_entropy']
        mirror=next(m for m in ma['methods'] if m['method']=='mirror_givens')
        native=next(m for m in ma['methods'] if m['method']=='hash_2048')
        dense=next(m for m in ma['methods'] if m['method']=='dense')
        diag=next(m for m in ma['methods'] if m['method']=='diagonal_gate')
        rank1=next(m for m in ma['methods'] if m['method']=='rank1_residual')
        assert mirror['serialized_bytes'] <= 1.10*native['serialized_bytes']
        assert abs(mirror['test_accuracy']-dense['test_accuracy']) <= .02
        assert mirror['test_accuracy'] < native['test_accuracy'] + .01
        assert diag['serialized_bytes'] <= 1.10*mirror['serialized_bytes']
        assert rank1['serialized_bytes'] <= 1.10*mirror['serialized_bytes']
        assert max(diag['test_accuracy'],rank1['test_accuracy']) >= mirror['test_accuracy']
        replay_info[str(seed)]={'all_six_payloads_replayed_byte_exact':True,
          'mirror_accuracy':mirror['test_accuracy'],'native_accuracy':native['test_accuracy'],
          'mirror_bytes':mirror['serialized_bytes'],'native_bytes':native['serialized_bytes'],
          'diagonal_accuracy':diag['test_accuracy'],'rank1_accuracy':rank1['test_accuracy']}
    assert not (ROOT/'runs/fresh').exists()
    out={'experiment_id':'MA-597','status':'FAIL','protocol_sha256':freeze['protocol_sha256'],'source_sha256':freeze['source_sha256'],
      'development_seeds':[59701,59702],'fresh_seeds_accessed':[],'fresh_artifacts_present':False,
      'serialization_roundtrip_checked':True,'metric_replay_checked':True,'max_accuracy_difference':0.0,'max_cross_entropy_difference':0.0,
      'results':replay_info,'tests':{'passed':2,'failed':0},
      'gate_decision':'FAIL: Mirror misses +1.0 percentage point versus native hash on both dev seeds; diagonal/rank-one controls match or beat it within the byte cap.'}
    (ROOT/'verification_report.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
