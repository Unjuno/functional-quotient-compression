import argparse,csv,hashlib,json
from pathlib import Path
def verify(root):
 p=Path(root)/'runs/a6/fresh'; rows=list(csv.DictReader((p/'RESULTS_CORE.csv').open())); assert len(rows)==15
 checked=0
 for seed in [56711,56712,56713]:
  d=p/str(seed); m=json.loads((d/'metrics.json').read_text()); assert m['selected_deep_fraction']==.5
  for k,n in m['object_bytes'].items():
   f=d/f'{k}.npz'; assert f.stat().st_size==n and hashlib.sha256(f.read_bytes()).hexdigest()==m['hashes'][k]
  import numpy as np
  with np.load(d/'shared.npz') as state: assert state['A'].dtype==np.float64 and state['difficulty_coeff'].dtype==np.float64
  assert len([r for r in rows if int(r['seed'])==seed])==5
  a={r['method']:r for r in rows if int(r['seed'])==seed}
  for r in a.values():
   assert abs(float(r['payload_bytes'])-m['total_payload_bytes'][r['method']])<1
   if r['method']!='mod_only': assert abs(float(r['nrmse']))<1e-4
   assert float(r['selected_deep_fraction'])==.5
  x,y=a['mod_mirror'],a['mod_direct_angle']
  assert x['nrmse']==y['nrmse'] and x['payload_bytes']==y['payload_bytes'] and x['active_macs']==y['active_macs']
  assert abs(float(a['dense_upper']['executed_deep_fraction'])-1.)<1e-9
  checked+=5
 return {'fresh_worlds':3,'rows':checked,'serialized_objects_verified':15,'direct_angle_alias_exact':True,'verified':True}
if __name__=='__main__':
 q=argparse.ArgumentParser();q.add_argument('--root',required=True);q.add_argument('--out');a=q.parse_args();r=verify(a.root);s=json.dumps(r,indent=2)+'\n';print(s,end='');a.out and Path(a.out).write_text(s)
