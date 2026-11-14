#!/usr/bin/env python3
"""Replay MA-1010's frozen synthetic development screen; fresh stays sealed."""
import hashlib,json,math,pathlib,subprocess,sys
E=pathlib.Path(__file__).resolve().parents[1];R=E/'runs';lock=json.loads((R/'development_lock.json').read_text())
subprocess.run([sys.executable,str(E/'source'/'d2sa_screen.py')],check=True,stdout=subprocess.DEVNULL)
a=json.loads((R/'development.json').read_text())
assert a['code_sha256']==lock['code_sha256']
assert a['seed_protocol']==lock['seed_protocol'] and a['seed_protocol']['fresh_opened'] is False
assert a['payload_bytes']==lock['payload_bytes']
maxdiff=0.
for method,m in lock['development_metrics'].items():
 for k,v in m.items():
  if k=='wall_s' or not isinstance(v,(int,float)):continue
  diff=abs(v-a['development_metrics'][method][k]); maxdiff=max(maxdiff,diff)
  assert math.isclose(v,a['development_metrics'][method][k],rel_tol=1e-10,abs_tol=1e-12),(method,k,v,a['development_metrics'][method][k])
cp=R/'development_models.pt';assert hashlib.sha256(cp.read_bytes()).hexdigest()==lock['checkpoint_sha256']
assert not list(R.glob('fresh*'))
(R/'development.json').write_text(json.dumps(lock,indent=2)+'\n')
print(json.dumps({'metric_replay_max_abs_difference':maxdiff,'payload_bytes_exact':True,'source_checkpoint_hash_exact':True,'fresh_opened':False}))
