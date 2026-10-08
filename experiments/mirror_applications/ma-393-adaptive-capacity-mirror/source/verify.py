import hashlib,json,sys,zipfile
from pathlib import Path
import numpy as np
from run import METHODS,evaluate,load,make_data
def check(root,file):
 root=Path(root);r=json.loads(Path(file).read_text());data=make_data(r['seed']);out=[]
 for row in r['summaries']:
  p=root/'payloads'/f"development_{r['seed']}_{row['method']}.npz";raw=p.read_bytes()
  with zipfile.ZipFile(p) as z:a={n[:-4]:np.load(z.open(n),allow_pickle=False) for n in z.namelist()}
  m=load(row['method'],r['seed'],a);replay={s:evaluate(m,data[s]) for s in ('validation','test')}
  out.append({'method':row['method'],'bytes_match':len(raw)==row['serialized_bytes'],'sha256_match':hashlib.sha256(raw).hexdigest()==row['payload_sha256'],'metadata_match':a['meta'].tolist()==[128,32,96,16,8,METHODS.index(row['method'])],'finite':all(np.isfinite(v).all() for v in a.values()),'metrics_match':all(abs(replay[s][k]-row[s][k])<1e-7 for s in replay for k in replay[s])})
 return {'seed':r['seed'],'checks':out,'passed':all(all(v for k,v in x.items() if k!='method') for x in out)}
if __name__=='__main__':
 r=check(sys.argv[1],sys.argv[2]);print(json.dumps(r,indent=2));raise SystemExit(not r['passed'])
