import importlib.util
from pathlib import Path

import torch

SRC=Path(__file__).resolve().parents[1]/'source'/'run.py'
spec=importlib.util.spec_from_file_location('ma332_run',SRC);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_hidden_permutation_preserves_exact_function():
    model,x,_=m.world(33201);ref=m.forward(x,*model)
    for i in range(8):
        p=torch.randperm(m.HIDDEN,generator=torch.Generator().manual_seed(i))
        out=m.forward(x,*m.transformed(model,p))
        assert torch.max(torch.abs(ref-out)) <= 1e-5

def test_incoming_only_permutation_changes_function():
    model,x,_=m.world(33202);ref=m.forward(x,*model);q=m.transformed(model,torch.arange(m.HIDDEN-1,-1,-1));q[2]=model[2]
    assert not torch.allclose(ref,m.forward(x,*q))

def test_payload_archive_roundtrip(tmp_path):
    model,_,_=m.world(33203);arrays=m.arrays_for(model);n,h=m.pack(arrays,tmp_path/'p.zip')
    got=m.load(tmp_path/'p.zip');assert n==(tmp_path/'p.zip').stat().st_size and len(h)==64
    for k in arrays:assert (arrays[k]==got[k]).all()

def test_shared_permutation_payload_is_compact(tmp_path):
    model,_,_=m.world(33204);paths=[];rng=__import__('numpy').random.default_rng(1)
    for i in range(8):
        n,_=m.pack(m.arrays_for(m.transformed(model,rng.permutation(m.HIDDEN))),tmp_path/f'{i}.zip');paths.append(n)
    shared={**m.arrays_for(model),'permutations':rng.integers(0,m.HIDDEN,(8,m.HIDDEN),dtype='u1')}
    n,_=m.pack(shared,tmp_path/'shared.zip');assert n<sum(paths)
