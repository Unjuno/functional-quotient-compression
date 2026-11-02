import argparse,csv,hashlib,json
from pathlib import Path
def verify(root):
 p=Path(root)/'runs/a1/fresh';rows=list(csv.DictReader((p/'RESULTS_CORE.csv').open()));assert len(rows)==60
 for s in [56911,56912,56913]:
  d=p/str(s);m=json.loads((d/'metrics.json').read_text())
  for k,n in m['object_bytes'].items():
   f=d/f'{k}.npz';assert f.stat().st_size==n and hashlib.sha256(f.read_bytes()).hexdigest()==m['hashes'][k]
  part=[r for r in rows if int(r['seed'])==s]; assert len(part)==20
  for step in [2,3,5,6]:
   a={r['method']:r for r in part if int(r['step'])==step};x,y=a['mirror'],a['direct_time']
   assert x['nrmse']==y['nrmse'] and x['payload_bytes']==y['payload_bytes'] and x['active_macs']==y['active_macs']
 return {'fresh_worlds':3,'rows':60,'serialized_objects':15,'direct_alias_exact':True,'verified':True}
if __name__=='__main__':
 q=argparse.ArgumentParser();q.add_argument('--root',required=True);q.add_argument('--out');a=q.parse_args();r=verify(a.root);s=json.dumps(r,indent=2)+'\n';print(s,end='');a.out and Path(a.out).write_text(s)
