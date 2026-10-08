import json,sys,time,zipfile
from pathlib import Path
import numpy as np,torch
from run import load_model

def bench(root,result_path,repeats=100):
 root=Path(root);result=json.loads(Path(result_path).read_text());token=torch.arange(64).repeat_interleave(4).repeat(64);domain=torch.arange(4).repeat(64*64);out=[]
 for row in result['summaries']:
  path=root/'payloads'/f"development_{result['seed']}_{row['method']}.npz"
  with zipfile.ZipFile(path) as z: arrays={n[:-4]:np.load(z.open(n),allow_pickle=False) for n in z.namelist()}
  model=load_model(row['method'],result['seed'],arrays)
  with torch.no_grad():
   model(token,domain);start=time.perf_counter()
   for _ in range(repeats):model(token,domain)
   elapsed=time.perf_counter()-start
  count=repeats*len(token);out.append({'method':row['method'],'examples':count,'repeats':repeats,'wall_seconds':elapsed,'examples_per_second':count/elapsed})
 return out
if __name__=='__main__':print(json.dumps(bench(sys.argv[1],sys.argv[2]),indent=2))
