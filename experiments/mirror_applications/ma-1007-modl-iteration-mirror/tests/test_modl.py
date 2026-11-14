import pathlib,sys,torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'source'))
from modl_torch import make_case,forward_op,adjoint_op,MoDLNet

def test_forward_adjoint_operator():
 q=make_case(107001,'A');x=torch.randn(32,32,dtype=torch.complex64);y=torch.randn_like(q['y'])
 lhs=(forward_op(x,q['sens'],q['mask']).conj()*y).sum();rhs=(x.conj()*adjoint_op(y,q['sens'],q['mask'])).sum()
 assert torch.allclose(lhs,rhs,atol=2e-4,rtol=2e-4)
def test_mirror_parameter_is_three_angles_and_preserves_hidden_pair_norm():
 m=MoDLNet('mirror');assert m.angles.numel()==3
 h=torch.randn(1,8,32,32);angle=m.angles[0];pair=h.reshape(1,4,2,32,32);c=torch.cos(angle);s=torch.sin(angle)
 out=torch.stack((c*pair[:,:,0]-s*pair[:,:,1],s*pair[:,:,0]+c*pair[:,:,1]),dim=2).reshape_as(h)
 assert torch.allclose(h.square().sum(1),out.square().sum(1),atol=1e-5)
def test_shared_mirror_has_same_denoiser_count_as_tied_modl():
 tied=MoDLNet('tied');view=MoDLNet('mirror');assert sum(p.numel() for p in tied.denoiser.parameters())==sum(p.numel() for p in view.denoiser.parameters())
 assert sum(p.numel() for p in view.angles)==3
def test_independent_and_tied_share_first_denoiser_initialization():
 torch.manual_seed(107001);a=MoDLNet('independent')
 torch.manual_seed(107001);b=MoDLNet('tied')
 assert torch.equal(a.denoisers[0].conv1.weight,b.denoiser.conv1.weight)
 assert torch.equal(a.denoisers[0].conv3.weight,b.denoiser.conv3.weight)
