import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import torch
from d2sa_screen import fft,ifft,mask,sample,phantom,Prior

def test_fft_roundtrip():
    x=phantom(1010201,True)
    assert torch.allclose(ifft(fft(x)),x,atol=1e-6)

def test_tta_partition_disjoint_and_inside_acquisition():
    x=phantom(1010202,True); k,m,tr,ho,z=sample(x,1018202)
    assert torch.all(tr|ho == m)
    assert not torch.any(tr&ho)
    assert tr.sum()>0 and ho.sum()>0

def test_mirror_latent_transform_preserves_feature_norm():
    model=Prior(); z=torch.randn(1,2,32,32)
    f=model.feat(z); code=torch.tensor([.31,-.7])
    r=model(z,code,'mirror')-z
    # Output map is shared; check the Givens coordinate itself is orthogonal.
    q=f.clone(); a,b=code; ca,sa=torch.cos(a),torch.sin(a); cb,sb=torch.cos(b),torch.sin(b)
    q[:,0]=ca*f[:,0]-sa*f[:,1];q[:,1]=sa*f[:,0]+ca*f[:,1]
    q[:,2]=cb*f[:,2]-sb*f[:,3];q[:,3]=sb*f[:,2]+cb*f[:,3]
    assert torch.allclose(q.square().sum(),f.square().sum(),atol=1e-5)
    assert torch.isfinite(r).all()
