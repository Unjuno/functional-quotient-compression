import sys
from pathlib import Path
import numpy as np
import torch

SRC=Path(__file__).resolve().parents[1]/'source'
sys.path.insert(0,str(SRC))
from run_screen import TaskHeads, metric_forecast, metric_class


def test_mirror_identity_and_norm_preserving_rotation():
    m=TaskHeads('mirror')
    x=torch.randn(2,1,64,768)
    y=m.transform(x,0)
    assert torch.equal(y,x)
    with torch.no_grad(): m.codes[0,0]=0.2
    z=m.transform(x,0)
    assert not torch.allclose(z,x)
    assert torch.allclose(z.norm(dim=-1),x.norm(dim=-1),atol=2e-5)


def test_task_output_shapes_and_equal_code_bytes():
    models=[TaskHeads('shared'),TaskHeads('diag'),TaskHeads('mirror'),TaskHeads('shared',independent=True)]
    f=torch.randn(3,1,64,768); a=f.clone(); c=torch.randn(3,768)
    assert models[0].forward_task(f,0).shape==(3,96)
    assert models[0].forward_task(a,1).shape==(3,1,64,8)
    assert models[0].forward_task(c,2).shape==(3,5)
    assert models[1].codes.numel()==models[2].codes.numel()==24
    assert models[1].codes.shape==(3,8)


def test_task_metrics_are_separate_and_reversible():
    meta={'y_raw':[[[1.,2.]],[[3.,4.]]],'window_mean':[2.,3.],'window_std':[2.,2.],'mase_scale':1.}
    # normalized predictions restore to raw targets exactly.
    p=np.array([[[-.5,0.]],[[0.,.5]]],np.float32)
    assert metric_forecast(np.zeros_like(p),p,meta)['mae']==0.0
    assert metric_class(np.array([0,1]),np.array([[2.,1.],[1.,2.]]))['accuracy']==1.0
