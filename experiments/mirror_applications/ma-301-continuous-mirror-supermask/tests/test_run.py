import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run import Model,serialize,unpack_payload,world

def test_payload_roundtrip_all_methods():
    for i,k in enumerate(['tied','binary_mask','continuous_gate','mirror','independent']):
        m=Model(k,10+i); blob=serialize(m,77); v=unpack_payload(blob)
        assert v.shape==(32,32) and len(blob)>0
        assert torch.isfinite(v).all()
        assert torch.allclose(v,m.vectors().detach(),atol=0.005,rtol=1e-3)
    assert len(serialize(Model('mirror',41),77)) < len(serialize(Model('binary_mask',41),77))

def test_mirror_coordinate_changes_function():
    m=Model('mirror',20);a=m.vectors().clone()
    with torch.no_grad():m.theta[0]=.2
    b=m.vectors();assert not torch.equal(a[0],b[0])

def test_feature_splits_are_separate():
    _,h,y,hv,yv,ht,yt=world(30301)
    assert h.shape==(2048,32) and hv.shape==(1024,32) and ht.shape==(2048,32)
    assert not torch.equal(h[:1024],hv[:1024]) and not torch.equal(h[:1024],ht[:1024])
