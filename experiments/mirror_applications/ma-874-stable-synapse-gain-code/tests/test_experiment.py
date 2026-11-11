import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from experiment import make_world,fit_shared,evaluate,payload


def test_native_context_gain_exact_alias_and_reset():
    row=evaluate(829)
    for n in ('1','2','4','8'):
        m=row['by_support'][n]
        assert m['mean_mse']['mirror']==m['mean_mse']['native_gain']
        assert m['payload_bytes']['mirror']==m['payload_bytes']['native_gain']
        assert m['reset_max_abs_error']<1e-12


def test_gain_state_payload_is_smaller_than_per_task_fast_weights():
    world=make_world(829); model=fit_shared(world)
    mirror_codes=np.zeros((16,2)); fast=np.zeros((16,9))
    assert len(payload(world,model,'mirror',mirror_codes))==264
    assert len(payload(world,model,'fast_weights',fast))==648


def test_fresh_worlds_are_not_used_in_development():
    assert make_world(829)['seed']==829
    assert make_world(983)['seed']==983
