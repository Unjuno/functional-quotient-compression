import json, sys, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run

ROOT=Path(__file__).resolve().parents[1]
records=json.loads((ROOT/'source/fresh_metrics.json').read_text())
checks=[]; max_diff=0.0; hashes={}
for seed in (29611,29612,29613):
 for condition in ('aligned_orthogonal_orbit','independent_isotropic_deltas'):
  replay=run.run_world(seed,condition,svd_rank=8,psp_seed=296103)
  saved=[r for r in records if r['world_or_seed']==seed and r['condition']==condition]
  assert len(replay)==len(saved)==6
  for got,want in zip(replay,saved):
   assert got['method']==want['method']
   assert got['serialized_bytes']==want['serialized_bytes']
   for key in ('task_output_norm_mse','signed_pair_norm_mse'):
    diff=abs(got[key]-want[key]); max_diff=max(max_diff,diff); assert diff<1e-14,(seed,condition,got['method'],key,diff)
   package=run.package(got['method'],run.world(seed,condition),rank=8,psp_seed=296103)
   hashes[f'{seed}:{condition}:{got["method"]}']=hashlib.sha256(run.serialize(package)).hexdigest()
   checks.append({'seed':seed,'condition':condition,'method':got['method'],'bytes':got['serialized_bytes'],'metric_replay_max_abs_diff':max(abs(got['task_output_norm_mse']-want['task_output_norm_mse']),abs(got['signed_pair_norm_mse']-want['signed_pair_norm_mse']))})
(ROOT/'source/replay_verification.json').write_text(json.dumps({'rows_replayed':len(checks),'max_metric_difference':max_diff,'payload_sha256':hashes,'checks':checks},indent=2)+'\n')
print(f'replayed {len(checks)} rows; max metric difference {max_diff:.3g}; payload hashes recorded')
