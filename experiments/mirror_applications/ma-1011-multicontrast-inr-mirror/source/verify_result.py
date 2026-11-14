#!/usr/bin/env python3
import hashlib,json,math,pathlib,subprocess,sys
E=pathlib.Path(__file__).resolve().parents[1];R=E/'runs';lock=json.loads((R/'development_lock.json').read_text())
subprocess.run([sys.executable,str(E/'source'/'multicontrast_screen.py')],check=True,stdout=subprocess.DEVNULL)
a=json.loads((R/'development.json').read_text());assert a['code_sha256']==lock['code_sha256'];assert a['fresh_opened'] is False and a['fresh_seeds']==lock['fresh_seeds'];assert a['payload_bytes']==lock['payload_bytes'];mx=0.
for name,metrics in lock['results'].items():
 for k,v in metrics.items():
  if k=='wall_s':continue
  d=abs(v-a['results'][name][k]);mx=max(mx,d);assert math.isclose(v,a['results'][name][k],rel_tol=1e-10,abs_tol=1e-12),(name,k,d)
assert hashlib.sha256((R/'development_models.pt').read_bytes()).hexdigest()==lock['checkpoint_sha256'];assert not list(R.glob('fresh*'))
(R/'development.json').write_text(json.dumps(lock,indent=2)+'\n')
print(json.dumps({'metric_replay_max_abs_difference':mx,'payload_bytes_exact':True,'checkpoint_hash_exact':True,'fresh_opened':False}))
