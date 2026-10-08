import csv,json,math,sys
from pathlib import Path
import torch
from model import METHODS,compute_proxy
from run import D,ROLES,UPDATES,BATCH,evaluate,fit,world

def main(core,exploratory,devgrid):
 torch.set_num_threads(1);rows=[]
 for path in (core,exploratory,devgrid):rows.extend(csv.DictReader(open(path,newline='')))
 seen={};unique=[]
 for row in rows:
  key=(row['seed'],row['condition'],row['method'],row['learning_rate'])
  if key in seen:
   for f in ('serialized_bytes','activation_mse','r2','active_compute_proxy'):assert row[f]==seen[key][f],(key,f)
   continue
  seen[key]=row;unique.append(row)
 maxm=maxr=0.
 for row in unique:
  seed=int(row['seed']);c=row['condition'];ci=0 if c=='aligned_block_shift' else 1;mi=METHODS.index(row['method']);lr=float(row['learning_rate']);w=world(seed,c);ds=seed+10000+ci*5000
  m,_=fit(row['method'],w,seed+mi*73+ci*10000,ds,lr);mse,r2,_,_=evaluate(m,w,seed+91+ci*3000)
  assert int(row['serialized_bytes'])==m.serialized_payload_bytes()
  assert int(row['active_compute_proxy'])==compute_proxy(row['method'],UPDATES*BATCH)
  assert int(row['train_examples'])==UPDATES*BATCH and int(row['optimizer_updates'])==UPDATES
  dm=abs(float(row['activation_mse'])-mse);dr=abs(float(row['r2'])-r2);maxm=max(maxm,dm);maxr=max(maxr,dr)
  assert math.isclose(float(row['activation_mse']),mse,rel_tol=1e-6,abs_tol=1e-10)
  assert math.isclose(float(row['r2']),r2,rel_tol=1e-6,abs_tol=1e-8)
 print(json.dumps({'experiment_id':'MA-181','report_records':len(rows),'distinct_training_rows_replayed':len(unique),'payload_bytes_and_compute_exact':True,'max_mse_difference':maxm,'max_r2_difference':maxr},indent=2))
if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
