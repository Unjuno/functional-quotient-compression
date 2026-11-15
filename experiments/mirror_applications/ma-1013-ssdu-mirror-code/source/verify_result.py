#!/usr/bin/env python3
"""Replay the fixed SSDU synthetic scan-code experiment; never opens fresh scans."""
import hashlib,json,math,pathlib,subprocess,sys
E=pathlib.Path(__file__).resolve().parents[1];R=E/'runs';lock=json.loads((R/'development_lock.json').read_text());subprocess.run([sys.executable,str(E/'source'/'ssdu_screen.py')],check=True,stdout=subprocess.DEVNULL);a=json.loads((R/'development.json').read_text());assert a['code_sha256']==lock['code_sha256'];assert a['fresh_seeds']==lock['fresh_seeds'] and not a['fresh_opened'];assert a['payload_bytes_three_scans']==lock['payload_bytes_three_scans'] and a['payload_bytes_one_scan']==lock['payload_bytes_one_scan'];mx=0.
for name,m in lock['results'].items():
 for k,v in m.items():
  if k=='wall_s':continue
  diff=abs(v-a['results'][name][k]);mx=max(mx,diff);assert math.isclose(v,a['results'][name][k],rel_tol=1e-10,abs_tol=1e-12),(name,k,diff)
assert hashlib.sha256((R/'development_models.pt').read_bytes()).hexdigest()==lock['checkpoint_sha256'];assert not list(R.glob('fresh*'));(R/'development.json').write_text(json.dumps(lock,indent=2)+'\n');print(json.dumps({'metric_replay_max_abs_difference':mx,'payload_bytes_exact':True,'checkpoint_hash_exact':True,'fresh_opened':False}))
