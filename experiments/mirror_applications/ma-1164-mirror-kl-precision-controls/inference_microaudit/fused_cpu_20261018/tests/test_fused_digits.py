import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from numba import set_num_threads
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import os
core_source=Path(__file__).resolve().parents[3]/'source'
if not core_source.exists():core_source=Path(os.environ['MA1164_EXPERIMENT_SOURCE'])
sys.path.insert(0,str(core_source))
from core import CoeffDecoder,logits
from fused_digits import make_program, decode, infer

set_num_threads(1)
torch.set_num_threads(1)

@pytest.mark.parametrize('m,K,R',[(1,1,1),(32,2,2),(256,4,3),(768,8,4),(1210,3,3)])
def test_decode_compiled_identical_to_original(m,K,R):
    g=np.random.default_rng(112+m)
    indices=np.sort(g.choice(1210,size=m,replace=False))
    u=g.standard_normal(m)*.02
    old=CoeffDecoder(indices,'mirror',k=K,rounds=R)
    with torch.no_grad():
        old.coords.copy_(torch.from_numpy(u))
        target=old.weights().numpy()
        got=decode(u,indices,make_program(K,R))
        assert max(abs(got-target))<1e-12
        x=torch.from_numpy(g.standard_normal((4,64)))
        expected=logits(torch.from_numpy(target),x).numpy()
        pred=infer(u,indices,make_program(K,R),x.numpy())
        assert np.max(np.abs(expected-pred))<1e-12


def test_old_real_checkpoint_accuracy_preserved():
    ckpt=Path(os.environ.get('MA1163_COMPILED_CHECKPOINT','/tmp/nonexistent_fqc_checkpoint.pt'))
    if not ckpt.exists():pytest.skip('Optional prior-trained MA1163 checkpoint not supplied')
    obj=torch.load(ckpt,weights_only=True,map_location='cpu')
    assert obj['m']==1210
    u=obj['state']['coords'].numpy()
    indices=np.arange(1210)
    k=3;r=3
    core=CoeffDecoder(indices,kind='mirror',k=k,rounds=r)
    with torch.no_grad():core.coords.copy_(torch.from_numpy(u))
    x=torch.from_numpy(np.random.default_rng(19).normal(size=(64,64)))
    with torch.no_grad(): ref=core(x).numpy()
    pred=infer(u,indices,make_program(k,r),x.numpy())
    assert np.max(np.abs(pred-ref))<1e-12
    assert np.array_equal(pred.argmax(-1),ref.argmax(-1))
