#!/usr/bin/env python3
"""Replay MA-1008 development deterministically; do not touch fresh seeds."""
import hashlib, json, math, os, pathlib, subprocess, sys, torch

ROOT=pathlib.Path(__file__).resolve().parents[4]
EXP=pathlib.Path(__file__).resolve().parents[1]
RUNS=EXP/'runs'
lock=json.loads((RUNS/'development_lock.json').read_text())
env=os.environ.copy(); env['PYTHONPATH']='/tmp/fastmri_official'+os.pathsep+env.get('PYTHONPATH','')
subprocess.run([sys.executable,str(EXP/'source'/'run_screen.py')],check=True,env=env,stdout=subprocess.DEVNULL)
actual=json.loads((RUNS/'development.json').read_text())
assert actual['code_sha256']==lock['code_sha256']
assert actual['fresh_opened'] is False and actual['fresh_seeds']==[1008301,1008307,1008313]
assert actual['source_pairs']==lock['source_pairs']
maxdiff=0.0
for method, metrics in lock['metrics'].items():
    for key in ('nmse','psnr_db','measured_coil_residual'):
        delta=abs(metrics[key]-actual['metrics'][method][key]);maxdiff=max(maxdiff,delta)
        assert math.isclose(metrics[key],actual['metrics'][method][key],rel_tol=0,abs_tol=0)
for section in ('factorized_givens_mirror','factorized_scalar_film','hard_shared_frozen','direct_pair_givens_target_adapt','independent_target_finetune'):
    assert lock['payloads'][section]['bytes']==actual['payloads'][section]['bytes']
assert lock['bank_storage_comparison']==actual['bank_storage_comparison']
cp=RUNS/actual['checkpoint']; blob=torch.load(cp,map_location='cpu',weights_only=False)
buf=__import__('io').BytesIO();torch.save(blob,buf);buf.seek(0);again=torch.load(buf,map_location='cpu',weights_only=False)
for key in blob:
    a,b=blob[key],again[key]
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for name in a: assert torch.equal(a[name],b[name])
    else: assert torch.equal(a,b)
assert hashlib.sha256(cp.read_bytes()).hexdigest()==actual['checkpoint_sha256']
assert not list(RUNS.glob('fresh*.json'))
print(json.dumps({'metric_replay_max_abs_difference':maxdiff,'payload_hashes_match':True,'checkpoint_roundtrip_exact':True,'fresh_opened':False}))
