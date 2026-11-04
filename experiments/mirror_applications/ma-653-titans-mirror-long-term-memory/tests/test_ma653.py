import json
from pathlib import Path
import numpy as np
from run_experiment import ROOT,make_world,evaluate

def test_rows_and_actual_complete_longterm_cache_bytes():
 rows=json.loads((ROOT/'source'/'development.json').read_text());assert len(rows)==40
 for x in rows:
  prefix=f"{x['seed']}_r{x['private_value_rank']}_{x['method']}"
  assert (ROOT/'source'/f'{prefix}_complete.npz').stat().st_size==x['complete_state_bytes']
  assert (ROOT/'source'/f'{prefix}_longterm.npz').stat().st_size==x['longterm_state_bytes']
  assert (ROOT/'source'/f'{prefix}_cache.npz').stat().st_size==x['shortterm_cache_bytes']

def test_mirror_native_lowrank_exact_alias():
 for seed in (65301,65302):
  for r in (0,2,4,8):
   a=ROOT/'source'/f'{seed}_r{r}_mirror_shared_private_complete.npz';b=ROOT/'source'/f'{seed}_r{r}_native_lowrank_memory_complete.npz'
   with np.load(a) as x,np.load(b) as y:
    for k in ('base_initialization','key_ids','surprise_threshold','surprise_write_total','shared_value_basis','key_codes','private_basis','private_codes','cache_keys','cache_values'):
     assert np.array_equal(x[k],y[k])

def test_metric_replay_from_serialized_memory():
 rows=json.loads((ROOT/'source'/'development.json').read_text())
 for x in rows:
  if x['method'] not in ('titans_full_matrix','shared_only_rank4','mirror_shared_private','explicit_key_value_table'):continue
  base,shared,private,values,events=make_world(x['seed'],x['private_value_rank'])
  recent=[];seen=set()
  for k,v in reversed(events):
   if k not in seen:recent.append((k,v));seen.add(k)
   if len(recent)==8:break
  recent=list(reversed(recent));cache_keys=np.array([k for k,_ in recent],dtype=np.uint8);cache_values=np.stack([v for _,v in recent])
  p=ROOT/'source'/f"{x['seed']}_r{x['private_value_rank']}_{x['method']}_complete.npz"
  with np.load(p) as data:
   if x['method']=='titans_full_matrix':long=data['memory_matrix']
   elif x['method']=='shared_only_rank4':long=data['key_codes']@data['shared_value_basis'].T
   elif x['method']=='mirror_shared_private':long=data['key_codes']@data['shared_value_basis'].T+data['private_codes']@data['private_basis'].T
   else:long=data['key_value_table']
   assert np.array_equal(data['cache_keys'],cache_keys)
   assert np.array_equal(data['cache_values'],cache_values)
  lm,aware,nll=evaluate(long,values,cache_keys,cache_values)
  assert abs(lm-x['longterm_query_mse'])<1e-10
  assert abs(aware-x['cache_aware_query_mse'])<1e-10
  assert abs(nll-x['cache_aware_query_nll'])<1e-8
