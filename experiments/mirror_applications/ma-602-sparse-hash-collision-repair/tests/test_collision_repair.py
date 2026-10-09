import sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run_collision_repair import hash_map,collision_positions,exception_indices,givens,METHODS,INPUT,HIDDEN

def test_collision_selection_matches_collision_excess():
 ix,sg=hash_map(2048,1234);pos=collision_positions(ix)
 assert len(pos)==INPUT*HIDDEN-len(np.unique(ix))
 assert np.all(pos>=0) and np.all(pos<INPUT*HIDDEN)

def test_random_exception_control_is_exactly_matched_and_deterministic():
 ix,_=hash_map(2048,60201);a=exception_indices('collision_residual_givens',ix,60201);b=exception_indices('random_residual_givens',ix,60201);c=exception_indices('random_residual_givens',ix,60201)
 assert len(a)==len(b) and np.array_equal(b,c)

def test_givens_preserves_norm_and_method_controls_exist():
 torch.manual_seed(1);x=torch.randn(4,INPUT);ang=torch.randn(INPUT//2)
 assert torch.allclose(x.norm(dim=-1),givens(x,ang).norm(dim=-1),atol=2e-6,rtol=2e-6)
 assert 'collision_residual_givens' in METHODS and 'hash_4096' in METHODS
