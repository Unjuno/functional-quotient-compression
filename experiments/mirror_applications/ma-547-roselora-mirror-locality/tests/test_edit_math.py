import sys
from pathlib import Path
import numpy as np
import torch

SOURCE=Path(__file__).resolve().parents[1]/'source'
sys.path.insert(0,str(SOURCE))
from run_experiment import mirror_edit_vector,row_edit_delta,route_ids


def test_mirror_vector_adds_fixed_target_old_margin():
    torch.manual_seed(1)
    a=torch.randn(512);b=torch.randn(512)
    m=mirror_edit_vector(a,b,2.0)
    assert abs(float((a-b)@m)-2.0)<1e-5


def test_rank_one_row_edit_hits_support_mean_margin():
    torch.manual_seed(2)
    h=torch.randn(512);delta=row_edit_delta(h,2.0)
    assert abs(float(h@delta)-2.0)<1e-5


def test_router_matches_only_exact_key_token_sequence():
    keys=[[4,5,6],[9,10]]
    assert route_ids([1,4,5,6,2],keys)==0
    assert route_ids([8,9,10,3],keys)==1
    assert route_ids([1,4,5,7,2],keys)==-1
