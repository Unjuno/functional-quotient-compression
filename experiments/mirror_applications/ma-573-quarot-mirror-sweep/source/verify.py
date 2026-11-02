import argparse,csv,hashlib,json
from pathlib import Path
def verify(root):
 p=Path(root)/'runs/a1/fresh';rows=list(csv.DictReader((p/'RESULTS_CORE.csv').open()));assert len(rows)==15
 for seed in [57311,57312,57313]:
  d=p/str(seed);m=json.loads((d/'metrics.json').read_text())
  for name,h in m['hashes'].items():
   f=d/f'{name}.npz';assert hashlib.sha256(f.read_bytes()).hexdigest()==h
  part={r['method']:r for r in rows if int(r['seed'])==seed};a,b=part['mirror'],part['direct_angle']
  assert a['nrmse']==b['nrmse'] and a['payload_bytes']==b['payload_bytes']
  assert float(a['full_precision_gauge_max_abs_error'])<1e-12
  assert int(a['calibration_evals'])==0
 return {'worlds':3,'rows':15,'hashes':15,'exact_direct_alias':True,'verified':True}
if __name__=='__main__':
 q=argparse.ArgumentParser();q.add_argument('--root',required=True);q.add_argument('--out');a=q.parse_args();r=verify(a.root);s=json.dumps(r,indent=2)+'\n';print(s,end='');a.out and Path(a.out).write_text(s)
