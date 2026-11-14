#!/usr/bin/env python3
"""Deterministic development replay for MA-1009; fresh seeds remain untouched."""
import hashlib,json,math,os,pathlib,subprocess,sys,torch
EXP=pathlib.Path(__file__).resolve().parents[1];RUNS=EXP/'runs';lock=json.loads((RUNS/'development_lock.json').read_text())
env=os.environ.copy();subprocess.run([sys.executable,str(EXP/'source'/'dune_screen.py')],check=True,env=env,stdout=subprocess.DEVNULL)
actual=json.loads((RUNS/'development.json').read_text());assert actual['code_sha256']==lock['code_sha256']
assert actual['seed_protocol']['fresh_opened'] is False
assert actual['seed_protocol']==lock['seed_protocol']
maxdiff=0.0
for method,metrics in lock['metrics'].items():
    for key in ('nmse','psnr_db','global_ssim','sample_residual','image_reencoded_residual'):
        delta=abs(metrics[key]-actual['metrics'][method][key]);maxdiff=max(maxdiff,delta)
        assert math.isclose(metrics[key],actual['metrics'][method][key],rel_tol=1e-7,abs_tol=1e-9),(method,key,delta)
    assert actual['metrics'][method]['sample_residual']<=1e-5
assert lock['payloads']==actual['payloads']
assert lock['deployment_payloads']==actual['deployment_payloads']
cp=RUNS/'development_models.pt';obj=torch.load(cp,map_location='cpu',weights_only=False)
buf=__import__('io').BytesIO();torch.save(obj,buf);buf.seek(0);again=torch.load(buf,map_location='cpu',weights_only=False)
for name,state in obj.items():
    assert state.keys()==again[name].keys()
    for key,value in state.items():assert torch.equal(value,again[name][key])
assert hashlib.sha256(cp.read_bytes()).hexdigest()==actual['checkpoint_sha256']
assert not list(RUNS.glob('fresh*.json'))
print(json.dumps({'max_metric_replay_abs_difference':maxdiff,'payload_hashes_exact':True,'checkpoint_roundtrip_exact':True,'acquired_sample_residual_max':max(v['sample_residual'] for v in actual['metrics'].values()),'fresh_opened':False}))
