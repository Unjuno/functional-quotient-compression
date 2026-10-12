import pytest
import torch
from source.experiment import Model,METHODS
from source.audit import compile_model

@pytest.mark.parametrize('method',METHODS)
def test_folded_forward_equivalence(method):
    model=Model(method,32,1,8,29,.12,True).eval()
    compiled=compile_model(model)
    x=torch.randn(11,64)
    with torch.no_grad():assert torch.allclose(model(x),compiled(x),atol=3e-6,rtol=2e-5)

def test_compiled_storage_counts_dense_cache():
    model=Model('learned',32,1,8,29,.12,True)
    compiled=compile_model(model)
    assert compiled.up.numel()==8*32*32
    assert compiled.head.numel()==8*1*32

@pytest.mark.parametrize('method',['shared','multihead'])
def test_native_shared_ffn_is_not_duplicated_in_deployment(method):
    model=Model(method,32,1,8,29,.12,True)
    compiled=compile_model(model)
    assert compiled.up.numel()==32*32

@pytest.mark.parametrize('wrapped',[False,True])
def test_hash_lock_accepts_local_and_remote_formats(tmp_path,wrapped):
    import json
    from source import audit
    payload={'source/planes.py':'abc'}
    p=tmp_path/'lock.json';p.write_text(json.dumps({'hashes':payload} if wrapped else payload))
    assert audit.read_hash_lock(p)==payload
