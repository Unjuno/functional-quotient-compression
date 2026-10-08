import json,sys,time,zipfile
from pathlib import Path
import numpy as np,torch
from run import load
def bench(root,file,repeats=100):
 root=Path(root);r=json.loads(Path(file).read_text());tok=torch.arange(128).repeat(128);out=[]
 for row in r['summaries']:
  p=root/'payloads'/f"development_{r['seed']}_{row['method']}.npz"
  with zipfile.ZipFile(p) as z:a={n[:-4]:np.load(z.open(n),allow_pickle=False) for n in z.namelist()}
  m=load(row['method'],r['seed'],a)
  with torch.no_grad():m(tok);start=time.perf_counter();
  with torch.no_grad():
   for _ in range(repeats):m(tok)
  elapsed=time.perf_counter()-start;out.append({'method':row['method'],'examples':repeats*len(tok),'wall_seconds':elapsed,'examples_per_second':repeats*len(tok)/elapsed})
 return out
if __name__=='__main__':print(json.dumps(bench(sys.argv[1],sys.argv[2]),indent=2))
