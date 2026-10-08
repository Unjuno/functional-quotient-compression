import hashlib,json,sys,zipfile
from pathlib import Path
import numpy as np
from run import METHODS,evaluate,load_model,make_data

def verify(root,result_path):
 root=Path(root);result=json.loads(Path(result_path).read_text());data=make_data(result['seed']);checks=[]
 for row in result['summaries']:
  path=root/'payloads'/f"development_{result['seed']}_{row['method']}.npz";raw=path.read_bytes()
  with zipfile.ZipFile(path) as z: arrays={n[:-4]:np.load(z.open(n),allow_pickle=False) for n in z.namelist()}
  model=load_model(row['method'],result['seed'],arrays)
  replay={s:evaluate(model,data[s],data) for s in ('validation','test')}
  checks.append({'method':row['method'],'bytes_match':len(raw)==row['serialized_bytes'],'sha256_match':hashlib.sha256(raw).hexdigest()==row['payload_sha256'],'metadata_match':arrays['meta'].tolist()==[8,8,64,4,16,8,METHODS.index(row['method'])],'finite':all(np.isfinite(x).all() for x in arrays.values()),'metrics_match':all(abs(replay[s][k]-row[s][k])<1e-7 for s in replay for k in replay[s])})
 return {'seed':result['seed'],'condition':result['condition'],'payload_checks':checks,'passed':all(all(v for k,v in c.items() if k!='method') for c in checks)}
if __name__=='__main__':
 x=verify(sys.argv[1],sys.argv[2]);print(json.dumps(x,indent=2));raise SystemExit(not x['passed'])
