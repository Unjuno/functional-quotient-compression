import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import torch
from site_screen import A,AH,sens,mask,phantom,Prior

def test_sense_forward_adjoint():
    g=torch.Generator().manual_seed(9);x=torch.complex(torch.randn(32,32,generator=g),torch.randn(32,32,generator=g));q=torch.complex(torch.randn(2,32,32,generator=g),torch.randn(2,32,32,generator=g));m,_,_=mask(5,1);s=sens(1);ax=A(torch.stack([x.real,x.imag]),s,m)
    lhs=(ax.conj()*q*m[None]).sum();rhs=(x.conj()*AH(q*m[None],s)).sum()
    assert torch.allclose(lhs,rhs,atol=1e-4,rtol=1e-5)

def test_site_masks_split_and_under_sample():
    m,tr,ho=mask(1012201,1)
    assert torch.equal(tr|ho,m) and not torch.any(tr&ho)
    assert int(m.sum())==8*32 and int(ho.sum())==32 and int(tr.sum())==7*32

def test_site_coils_are_normalized_and_givens_is_norm_preserving():
    s=sens(2);assert torch.allclose(s.abs().square().sum(0),torch.ones(32,32),atol=1e-6)
    model=Prior();z=torch.randn(1,2,32,32);f=model.feat(z);a,b=torch.tensor(.2),torch.tensor(-.6);ca,sa=torch.cos(a),torch.sin(a);cb,sb=torch.cos(b),torch.sin(b);q=f.clone()
    q[:,0]=ca*f[:,0]-sa*f[:,1];q[:,1]=sa*f[:,0]+ca*f[:,1];q[:,2]=cb*f[:,2]-sb*f[:,3];q[:,3]=sb*f[:,2]+cb*f[:,3]
    assert torch.allclose(q.square().sum(),f.square().sum(),atol=1e-4)
