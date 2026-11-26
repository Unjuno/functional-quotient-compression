import importlib.util
from pathlib import Path
import numpy as np
import torch

SRC=Path(__file__).resolve().parents[1]/'source'/'train.py'
spec=importlib.util.spec_from_file_location('ma1063',SRC); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def test_dense_hash_is_deterministic_and_bounded():
    ids=np.array([1,17,3952],dtype=np.int64)
    a=m.dense_hash(ids,0); b=m.dense_hash(ids,0)
    assert np.array_equal(a,b) and a.shape==(3,m.K)
    assert float(a.min())>=-1 and float(a.max())<=1

def test_mirror_preserves_embedding_norm_at_initialization():
    torch.manual_seed(3); model=m.Model('dhe_mirror',6040,3952)
    model.hash_u=torch.randn(6041,m.K); model.hash_i=torch.randn(3953,m.K)
    ids=torch.tensor([1,8,100])
    with torch.no_grad():
        raw=model.du(model.hash_u[ids]); viewed=model.embed(ids,0)
    assert torch.allclose(raw.norm(dim=1),viewed.norm(dim=1),atol=1e-6)

def test_mirror_and_paired_diagonal_have_matched_coordinate_count():
    mirror=m.Model('dhe_mirror',6040,3952); diagonal=m.Model('dhe_diag',6040,3952)
    assert mirror.view.numel()==diagonal.view.numel()==32

def test_all_methods_produce_finite_logits():
    u=torch.tensor([1,2,3]); i=torch.tensor([4,5,6])
    for method in m.METHODS:
        model=m.Model(method,6040,3952)
        if method!='full_tables':
            model.hash_u=torch.randn(6041,m.K); model.hash_i=torch.randn(3953,m.K)
        y=model(u,i)
        assert y.shape==(3,) and torch.isfinite(y).all()
