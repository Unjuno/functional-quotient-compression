"""Offline CPU replay of MA-876 fresh translator metrics."""
import json,sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
from experiment import run_fresh,FRESH_SEEDS
from datasets.utils import logging as datasets_logging
datasets_logging.set_verbosity_error()
saved=json.loads((ROOT/'fresh_results.json').read_text())
replay=run_fresh(FRESH_SEEDS,32)
for a,b in zip(saved['rows'],replay['rows']):
 assert a['seed']==b['seed']
 for key in ('family_full_bytes','family_shared_bytes_by_rank','heldout_full_map_bytes','heldout_shared_marginal_bytes','prompt_rows'):
  assert a[key]==b[key],(a['seed'],key)
 x=a['rank_results']['32']; y=b['rank_results']['32']
 for key in ('native_nll','full_map_nll','compressed_nll','relative_cache_mse_full','relative_cache_mse_compressed'):
  assert x[key]==y[key],(a['seed'],key)
 assert math.isclose(x['compressed_vs_full_nll_delta'],y['compressed_nll_delta']-y['full_map_nll_delta'],rel_tol=0,abs_tol=1e-12)
 assert x['quality_gate_pass']==(x['compressed_vs_full_nll_delta']<=.05)
print('MA-876 fresh replay exact; rank32 quality gate failed in all three fresh worlds')
