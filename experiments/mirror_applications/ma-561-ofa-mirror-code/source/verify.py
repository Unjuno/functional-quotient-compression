import argparse,csv,hashlib,json
from pathlib import Path

def verify(root):
 root=Path(root); fresh=root/'runs/a1/fresh'; rows=list(csv.DictReader((fresh/'RESULTS_CORE.csv').open()))
 assert len(rows)==48, len(rows)
 checks=0
 for seed in [56111,56112,56113]:
  d=fresh/str(seed); meta=json.loads((d/'metrics.json').read_text())
  assert meta['seed']==seed and meta['split']=='fresh'
  for name,n in meta['bytes'].items():
   p=d/f'{name}.npz'; assert p.stat().st_size==n
   assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256'][name]
  part=[r for r in rows if int(r['seed'])==seed]
  assert len(part)==16
  for r in part:
   assert r['method'] in {'native','mirror','rank1','independent'}
   assert int(r['payload_bytes'])==meta['bytes']['shared']+meta['bytes'].get(r['method'],0)
   checks+=1
  for r in part:
   if r['method'] in {'mirror','independent'}: assert float(r['nrmse'])<1e-7
  for c in {(r['width'],r['depth'],r['kernel']) for r in part}:
   by={r['method']:r for r in part if (r['width'],r['depth'],r['kernel'])==c}
   assert float(by['native']['nrmse'])>1e-4
  assert meta['bytes']['shared']+meta['bytes']['mirror'] <= .5*(meta['bytes']['shared']+meta['bytes']['independent'])
 return {'fresh_worlds':3,'metric_rows_checked':checks,'byte_hashes_checked':12,'result':'verified'}

if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--out',type=Path); a=ap.parse_args(); out=verify(a.root); txt=json.dumps(out,indent=2)+'\n'; print(txt,end=''); a.out and a.out.write_text(txt)
