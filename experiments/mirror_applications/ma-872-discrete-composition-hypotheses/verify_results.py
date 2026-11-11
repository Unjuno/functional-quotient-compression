"""Deterministic fresh metric replay for MA-872."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
from experiment import run
saved=json.loads((ROOT/'fresh_results.json').read_text()); replay=run([941,953,971])
for key in ('mean_mse_by_support','max_world_mean_mse_by_support','max_task_mse_by_support','payload_bytes'):
 assert saved[key]==replay[key],key
for a,b in zip(saved['rows'],replay['rows']):
 assert a['seed']==b['seed']
 for n in ('1','2','4','8','16','32'):
  x,y=a['by_support'][n],b['by_support'][n]
  for key in ('mean_mse','worst_task_mse','per_task_mse','payload_bytes','indices','search_macs_per_task'):
   assert x[key]==y[key],(a['seed'],n,key)
  assert x['mean_mse']['mirror']==x['mean_mse']['native_codebook']
  assert x['indices']['mirror']==x['indices']['native_codebook']
print('MA-872 fresh replay exact; native codebook alias exact')
