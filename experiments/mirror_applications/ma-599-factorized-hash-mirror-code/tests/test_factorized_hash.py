import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run_factorized_hash import METHODS,hash_map,init_params,expert_w1,BUCKETS,INPUT,HIDDEN

def test_hash_maps_are_deterministic():
    a=hash_map(BUCKETS,9921);b=hash_map(BUCKETS,9921);c=hash_map(BUCKETS,9922)
    assert np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1])
    assert not np.array_equal(a[0],c[0])

def test_factorized_view_uses_one_physical_shared_hash_and_rank4_codes():
    p=init_params('factorized_hash_view',59901,5990101)
    assert p['theta'].shape==(1,BUCKETS)
    assert p['bucket_basis'].shape==(BUCKETS,4)
    assert p['expert_code'].shape==(4,4)
    assert len(METHODS)==7

def test_router_initialization_matched():
    a=init_params('factorized_hash_view',59901,5990101)
    b=init_params('salted_shared_hash',59901,5990102)
    assert torch.equal(a['router_w'],b['router_w'])
    assert torch.equal(a['router_b'],b['router_b'])
