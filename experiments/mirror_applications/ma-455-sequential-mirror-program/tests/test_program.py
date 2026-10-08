from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

def test_teacher_steps_are_noncommutative():
    _,m=run.teacher(45501)
    a,b=m[0],m[1]
    assert torch.linalg.norm(a@b-b@a).item()>1e-2

def test_givens_rotation_is_orthogonal():
    r=run.rotation(torch.tensor([.31,-.22,.43]))
    assert torch.max(torch.abs(r.T@r-torch.eye(3))).item()<1e-6

def test_native_conditioner_is_same_map_and_payload():
    mirror=run.init('mirror',45501);native=run.init('native_givens',45501)
    assert all(torch.equal(a,b) for a,b in zip(mirror,native))
    assert torch.equal(run.predict(torch.eye(3),'mirror',mirror),run.predict(torch.eye(3),'native_givens',native))
    assert run.serialize('mirror',mirror)==run.serialize('native_givens',native)

def test_reversed_order_changes_teacher_function():
    _,m=run.teacher(45501)
    assert torch.linalg.norm(m[2]@m[1]@m[0]-m[0]@m[1]@m[2]).item()>1e-2
