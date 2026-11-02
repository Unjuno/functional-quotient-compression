import argparse,csv,hashlib,json
from pathlib import Path

def verify(root):
 root=Path(root); p=root/'runs/fresh'; rows=list(csv.DictReader((p/'RESULTS_CORE.csv').open())); assert len(rows)==48
 checked=0
 for seed in [56211,56212,56213]:
  d=p/str(seed); m=json.loads((d/'metrics.json').read_text()); part=[r for r in rows if int(r['seed'])==seed]; assert len(part)==16
  for name,n in m['bytes'].items():
   f=d/f'{name}.npz'; assert f.stat().st_size==n; assert hashlib.sha256(f.read_bytes()).hexdigest()==m['hashes'][name]
  for c in {(r['width'],r['depth']) for r in part}:
   x={r['method']:r for r in part if (r['width'],r['depth'])==c}
   assert x['mirror']['nrmse']==x['direct']['nrmse'] and x['mirror']['payload_bytes']==x['direct']['payload_bytes']
   assert float(x['mirror']['nrmse'])<1e-4
  checked+=len(part)
 return {'fresh_worlds':3,'rows':checked,'payload_hashes':12,'direct_alias_exact':True}
if __name__=='__main__':
 a=argparse.ArgumentParser(); a.add_argument('--root',required=True); a.add_argument('--out'); z=a.parse_args(); d=verify(z.root); s=json.dumps(d,indent=2)+'\n'; print(s,end=''); z.out and Path(z.out).write_text(s)
