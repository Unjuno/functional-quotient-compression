import argparse,csv,hashlib,json
from pathlib import Path
def verify(root):
 p=Path(root)/'runs/a1/fresh';rows=list(csv.DictReader((p/'RESULTS_CORE.csv').open()));assert len(rows)==15
 for s in [57111,57112,57113]:
  d=p/str(s);m=json.loads((d/'metrics.json').read_text())
  for k,n in m['object_bytes'].items():
   f=d/f'{k}.npz';assert f.stat().st_size==n and hashlib.sha256(f.read_bytes()).hexdigest()==m['hashes'][k]
  a={r['method']:r for r in rows if int(r['seed'])==s};x,y=a['mirror'],a['direct']
  assert x['nrmse']==y['nrmse'] and x['payload_bytes']==y['payload_bytes'] and x['active_macs']==y['active_macs']
  assert float(x['max_fixed_point_residual'])<=1e-7
 return {'worlds':3,'rows':15,'payload_hashes':15,'direct_alias':True,'verified':True}
if __name__=='__main__':
 q=argparse.ArgumentParser();q.add_argument('--root',required=True);q.add_argument('--out');a=q.parse_args();r=verify(a.root);s=json.dumps(r,indent=2)+'\n';print(s,end='');a.out and Path(a.out).write_text(s)
