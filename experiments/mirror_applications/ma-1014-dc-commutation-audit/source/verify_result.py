"""Deterministic replay and artifact verification for MA-1014."""
import hashlib, json, math, pathlib, subprocess, sys
E=pathlib.Path(__file__).resolve().parents[1]; R=E/'runs'; lock=json.loads((R/'development_lock.json').read_text())
subprocess.run([sys.executable,str(E/'source'/'dc_audit.py')],check=True,stdout=subprocess.DEVNULL)
now=json.loads((R/'development.json').read_text())
assert now['code_sha256']==lock['code_sha256']
assert now['source_seeds']==lock['source_seeds'] and now['target_seeds']==lock['target_seeds']
assert now['fresh_opened'] is False
assert len(now['results'])==len(lock['results'])
for a,b in zip(now['results'],lock['results']):
    for k in ('site','method','payload_bytes','updates','eval_examples'): assert a[k]==b[k],(k,a[k],b[k])
    for k in ('acquired_residual','unacquired_kspace_rmse','nmse'):
        assert math.isclose(a[k],b[k],rel_tol=1e-10,abs_tol=1e-12),(a['method'],k,a[k],b[k])
digest=hashlib.sha256((R/'development_model.pt').read_bytes()).hexdigest(); assert digest==lock['checkpoint_sha256']
(R/'development.json').write_text(json.dumps(lock,indent=2)+'\n')
print(json.dumps({'metrics_replayed_exactly':True,'payload_bytes_exact':True,'checkpoint_sha256':digest,'fresh_opened':False}))
