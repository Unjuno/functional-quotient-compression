from pathlib import Path
import hashlib,json
import pandas as pd
import torch
from model import reload_payload
from experiment import split_world
from full_supervision import compute_metrics
ROOT=Path(__file__).parent

def run():
 torch.set_num_threads(1)
 lock=json.loads((ROOT/'LONG_FULL_LOCK.json').read_text())
 for n,s in lock['source_sha256'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==s,n
 df=pd.read_csv(ROOT/'results/LONG_FULL.csv');assert len(df)==36
 count=0;maxdiff=0.
 for (world,seed,method),grp in df.groupby(['world','seed','method']):
  row=grp.iloc[-1]; f=ROOT/'models'/f'long_{world}_{seed}_{method}.pt';b=f.read_bytes()
  assert hashlib.sha256(b).hexdigest()==row.model_sha256
  assert len(b)==int(row.inference_bytes)
  result=compute_metrics(reload_payload(b),split_world(int(world))['test'])
  for k,v in result.items():
   d=abs(v-float(row['test_'+k]));assert d<5e-7,(k,d)
   maxdiff=max(maxdiff,d)
  count+=1
 assert count==12
 q={'source_lock':'PASS','long_model_count':12,'payload_sha_bytes':'PASS','max_metric_replay_abs_diff':maxdiff}
 (ROOT/'results/LONG_AUDIT.json').write_text(json.dumps(q,indent=2)+'\n');print(q)
if __name__=='__main__':run()
