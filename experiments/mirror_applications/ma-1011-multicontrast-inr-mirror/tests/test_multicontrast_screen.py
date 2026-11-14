import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import torch
from multicontrast_screen import truth,masks,kfft,SharedINR,coords,payload

def test_complementary_masks_and_heldout_are_disjoint():
    m,tr,ho=masks()
    assert torch.equal(tr|ho,m)
    assert not torch.any(tr&ho)
    assert torch.all((m.sum((1,2))==6*24))
    assert int(m.any(dim=(0,1)).sum()) > 18  # union covers most phase-encode lines

def test_ir_truth_is_a_joint_spatial_contrast_signal():
    x=truth()
    assert x.shape==(8,24,24)
    assert not torch.allclose(x[0],x[-1])
    assert torch.isfinite(x).all()

def test_givens_view_is_orthogonal():
    model=SharedINR(); xy=coords(); z=model.features(xy);v=torch.tensor([.31,-.72]);a,b=v
    ca,sa=torch.cos(a),torch.sin(a);cb,sb=torch.cos(b),torch.sin(b);q=z.clone()
    q[:,0]=ca*z[:,0]-sa*z[:,1];q[:,1]=sa*z[:,0]+ca*z[:,1];q[:,2]=cb*z[:,2]-sb*z[:,3];q[:,3]=sb*z[:,2]+cb*z[:,3]
    assert torch.allclose(q.square().sum(),z.square().sum(),atol=1e-5)
    assert len(payload(model,'mirror',torch.zeros(8,2)))>0
