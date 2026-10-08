from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

def test_teacher_gate_tensor_is_cp_rank_two():
    _,g,_,_=run.world(46601)
    # The logits are rank two across task versus layer×component by construction.
    logits=torch.logit(g.clamp(1e-6,1-1e-6))
    assert torch.linalg.matrix_rank(logits.reshape(run.TASKS,-1),atol=1e-4).item()<=run.R

def test_native_cp_is_exact_same_map_and_payload():
    comp,_,_,_=run.world(46601);a=run.model('mirror',46601);b=run.model('native_cp',46601)
    for k in a:assert torch.equal(a[k],b[k])
    x=torch.randn(5,run.D)
    assert torch.equal(run.predict('mirror',a,comp,3,2,x),run.predict('native_cp',b,comp,3,2,x))
    assert run.serialize('mirror',a,comp)==run.serialize('native_cp',b,comp)

def test_three_component_functions_are_distinct_nonzero():
    comp,_,_,_=run.world(46601);x=torch.eye(run.D)
    out=run.components(x,comp,2)
    assert all(float(q.abs().max())>0 for q in out)
    assert not torch.equal(out[0],out[1]) and not torch.equal(out[1],out[2])
