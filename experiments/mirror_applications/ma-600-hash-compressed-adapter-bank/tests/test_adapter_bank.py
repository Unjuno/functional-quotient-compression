import sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run_adapter_bank import METHODS,hash_map,params,matrices,teacher,INP,OUT,TASKS,BUCKETS

def test_task_teachers_are_deterministic_and_rank_two():
 a,b=teacher(60001);c,d=teacher(60001)
 assert np.array_equal(a,c) and np.array_equal(b,d)
 assert a.shape==(TASKS,INP,2) and b.shape==(TASKS,2,OUT)

def test_shared_givens_hash_produces_per_task_matrices():
 p=params('mirror_givens_hash',60001);maps=[hash_map(177) for _ in range(TASKS)]
 w=matrices('mirror_givens_hash',p,maps)
 assert w.shape==(TASKS,INP,OUT)
 assert torch.isfinite(w).all()

def test_registry_controls_present():
 assert METHODS==('independent_dense','independent_lora','independent_hash','shared_salted_hash','mirror_givens_hash','vera_shared_basis','generic_shared_basis')
 assert BUCKETS==512
