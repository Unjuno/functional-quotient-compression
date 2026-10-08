import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
import run_experiment as exp

def test_rotation_preserves_classifier_weight_norm():
    w=torch.randn(exp.C,exp.D);a=torch.tensor(.7);r=exp.rotate(w,a)
    assert float((w.norm()-r.norm()).abs())<1e-5

def test_aligned_teachers_share_rotation_orbit():
    teachers,angles,_tr,_te=exp.make_world(26001,'aligned')
    recovered=exp.rotate(teachers[0].expand(exp.E,-1,-1),angles-angles[0])
    # The first teacher is the orbit anchor; later members remain in the same orbit.
    assert float((recovered[1:]-teachers[1:]).abs().max())<1e-5

def test_unrelated_world_has_no_angle_codes():
    _t,a,_tr,_te=exp.make_world(26001,'unrelated')
    assert a is None
