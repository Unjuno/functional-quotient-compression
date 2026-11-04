import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'));import run_experiment as run

def test_payload_replay_retrieval_scope_and_bytes():
 rows=json.loads((ROOT/'source/development.json').read_text());assert len(rows)==16
 for row in rows:
  seed=row['seed'];sigma=row['residual_rms'];world=run.make_world(seed);left,right,scope,labels,targets=run.queries_for(seed,world,sigma);path=ROOT/'source'/f"{seed}_noise{sigma:g}_{row['method']}.npz";assert path.stat().st_size==row['serialized_bytes']
  with np.load(path) as z:m=run.retrieve(row['method'],z,left,right,scope,labels,targets)
  assert abs(m['recall']-row['retrieval_recall'])<1e-12
  assert abs(m['edit_nrmse']-row['output_nrmse'])<1e-7
  assert abs(m['wrong_scope_rate']-row['wrong_scope_rate'])<1e-12
  assert abs(m['absent_false_activation_rate']-row['absent_false_activation_rate'])<1e-12
  assert abs(m['collision_rate']-row['collision_rate'])<1e-12
