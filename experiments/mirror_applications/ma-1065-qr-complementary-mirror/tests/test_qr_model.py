import importlib.util
from pathlib import Path
import torch

SRC=Path(__file__).resolve().parents[1]/'source'/'train.py'
spec=importlib.util.spec_from_file_location('ma1065',SRC); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def test_shared_quotient_remainder_pairs_are_complementary():
    ids=torch.arange(m.N)
    pairs=torch.stack((ids//101,ids%101),dim=1)
    assert len(torch.unique(pairs,dim=0))==m.N
    assert int(pairs[:,0].max())<99 and int(pairs[:,1].max())<101

def test_independent_qr_partitions_cover_each_role():
    user=torch.arange(m.NU); item=torch.arange(m.NI)
    assert len(torch.unique(torch.stack((user//79,user%79),dim=1),dim=0))==m.NU
    assert len(torch.unique(torch.stack((item//63,item%63),dim=1),dim=0))==m.NI

def test_mirror_is_identity_at_initialization_and_matches_diagonal_bytes():
    base=m.QRModel('shared_qr'); mirror=m.QRModel('shared_qr_mirror'); diag=m.QRModel('shared_qr_diag')
    state={k:v for k,v in base.state_dict().items() if k in mirror.state_dict()}
    mirror.load_state_dict(state,strict=False)
    state2={k:v for k,v in base.state_dict().items() if k in diag.state_dict()}
    diag.load_state_dict(state2,strict=False)
    u=torch.tensor([1,7,6040]); i=torch.tensor([2,17,3952])
    with torch.no_grad(): expected=base(u,i); actual=mirror(u,i)
    assert torch.allclose(expected,actual,atol=1e-7)
    assert mirror.view.numel()==diag.view.numel()==32

def test_all_registered_methods_return_finite_logits():
    u=torch.tensor([1,2,5]); i=torch.tensor([4,9,11])
    for method in m.METHODS:
        model=m.QRModel(method); out=model(u,i)
        assert out.shape==(3,) and torch.isfinite(out).all()

def test_hash_control_collision_count_is_explicit():
    c=m.collisions()
    assert c['shared_hash_200']==m.N-200
    assert c['shared_qr_mirror']==0
