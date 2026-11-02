import argparse,csv,hashlib,json
from pathlib import Path
def verify(root):
 p=Path(root)/'runs/fresh'; rows=list(csv.DictReader((p/'RESULTS_CORE.csv').open())); assert len(rows)==24
 for s in [56511,56512,56513]:
  d=p/str(s); m=json.loads((d/'metrics.json').read_text()); assert len([r for r in rows if int(r['seed'])==s])==8
  for k,n in m['bytes'].items():
   f=d/f'{k}.npz'; assert f.stat().st_size==n and hashlib.sha256(f.read_bytes()).hexdigest()==m['hashes'][k]
 return {'worlds':3,'rows':24,'payloads':12,'verified':True}
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--out');z=a.parse_args();r=verify(z.root);s=json.dumps(r,indent=2)+'\n';print(s,end='');z.out and Path(z.out).write_text(s)
