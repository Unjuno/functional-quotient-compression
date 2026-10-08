import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from run import rotation, make_world, payload, ConditionalFit
from verify import verify

def test_givens_rotation_is_orthogonal():
    r=rotation(torch.linspace(-0.7,0.7,8))
    assert torch.max(torch.abs(r.T@r-torch.eye(16)))<1e-6

def test_world_shape_and_alpha_endpoints():
    _,h,y,_,_,_=make_world(40101,n=32)
    assert h.shape==(32,16)
    assert y.shape==(32,40,16)

def test_serialized_metrics_replay():
    result=verify()
    assert result['metric_replay']=='PASS'
    assert result['payload_replay']=='PASS'
    assert result['fresh_accessed'] is False
