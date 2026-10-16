import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from model import *
from experiment import split_world,evaluate,fit_teacher,soft_bce,soft_kl

def test_digits_all_pairs_have_nonempty_positive_classes():
    x,y=split_world(79010)['train']
    assert y.shape[1]==9
    assert (y.mean(0)>0.10).all() and (y.mean(0)<0.70).all()

def test_heldout_positions_cover_all_factors():
    assert set(i//3 for i in HOLD)=={0,1,2}
    assert set(i%3 for i in HOLD)=={0,1,2}
    assert len(SEEN)==6

def test_teacher_models_no_weight_tying():
    t=Teachers()
    for w in [t.wg,t.wu,t.wd]:assert w.shape[0]==9
    assert t.wg.data_ptr()!=t.wu.data_ptr()

def test_all_student_methods_are_finite_and_correct_shape():
    x=torch.randn(6,64)
    for method in METHODS:
        out=Student(method)(x)
        assert out.shape==(6,9) and torch.isfinite(out).all(),method

def test_identity_rotation_and_gain():
    x=torch.randn(5,3,32)
    assert torch.allclose(givens(x,torch.zeros(3,12)),x,atol=1e-6)
    assert torch.allclose(gain(x,torch.zeros(3,12)),x,atol=1e-6)

def test_mirror_and_gain_have_exactly_matching_trainable_params():
    assert sum(p.numel() for p in Student('mirror').parameters())==sum(p.numel() for p in Student('gain').parameters())
    assert sum(p.numel() for p in Student('mirror').parameters())==sum(p.numel() for p in Student('pair_mirror').parameters())

def test_all_9_mirror_functions_collapse_at_zero_view():
    m=Student('mirror')
    out=m(torch.randn(7,64))
    assert torch.allclose(out[:,0]-m.gbo[0]-m.ubo[0],out[:,8]-m.gbo[2]-m.ubo[2],atol=1e-6)

def test_givens_can_send_gradients_to_both_axes():
    m=Student('mirror');out=m(torch.randn(12,64))
    (out[:,4].square().mean()).backward()
    assert m.gangle.grad[1].abs().sum()>1e-8
    assert m.uangle.grad[1].abs().sum()>1e-8

def test_serialization_roundtrip_exact():
    m=Student('mirror');x=torch.randn(7,64)
    p=payload(m);n=reload_payload(p)
    assert torch.equal(m(x),n(x))

def test_student_loss_only_sees_allowed_task_columns():
    p=torch.randn(4,9,requires_grad=True);y=torch.randn(4,9)
    out=soft_bce(p,y,SEEN)
    grad=torch.autograd.grad(out,p)[0]
    assert torch.all(grad[:,HOLD]==0)
    assert torch.all(grad[:,SEEN]!=0)

def test_teacher_kl_nonnegative():
    z=torch.randn(5,9)
    assert soft_kl(z/2,z,SEEN).item()>=-1e-7
    assert soft_kl(z/2,z,SEEN).item()<1e-6

def test_world_split_has_no_image_overlap():
    data=split_world(79010)
    sets=[set(map(tuple,data[s][0].tolist())) for s in ['train','val','test']]
    assert not(sets[0]&sets[1] or sets[1]&sets[2] or sets[0]&sets[2])

def test_pair_readout_charges_only_actually_used_weights():
    m=Student('pair_readout')
    assert not hasattr(m,'wd'), 'unused shared wd artificially enlarges native readout payload'
    assert m.wd_pair.shape==(9,32)

def test_unrestricted_independent_has_unrestricted_pair_output_bias():
    m=Student('independent')
    assert m.bo.shape==(9,), 'independent FFN must allow independent task-specific output bias'
    assert not hasattr(m,'gbo')
