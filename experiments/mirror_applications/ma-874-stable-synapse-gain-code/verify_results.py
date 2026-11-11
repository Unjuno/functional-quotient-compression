"""Deterministic metric replay for MA-874; time fields are excluded."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
from experiment import run
saved=json.loads((ROOT/'fresh_results.json').read_text()); replay=run([983,997,1009])
for k in ('mean_mse_by_support','max_world_mean_mse_by_support','max_task_mse_by_support','payload_bytes'):
 assert saved[k]==replay[k],k
for a,b in zip(saved['rows'],replay['rows']):
 assert a['seed']==b['seed']
 for n in ('1','2','4','8'):
  x,y=a['by_support'][n],b['by_support'][n]
  for k in ('mean_mse','worst_task_mse','per_task_mse','payload_bytes','reset_max_abs_error'):
   assert x[k]==y[k],(a['seed'],n,k)
  assert x['mean_mse']['mirror']==x['mean_mse']['native_gain']
print('MA-874 fresh metric replay exact; native gain alias and resets exact')
