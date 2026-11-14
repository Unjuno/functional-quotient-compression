import pathlib,sys,torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'source'))
from run_screen import GivensView,ScalarView
class Echo(torch.nn.Module):
 def forward(self,x):return x

def test_givens_preserves_complex_norm():
 x=torch.randn(2,3,4,5,2);m=GivensView(Echo());m.angle.data.fill_(.63);y=m(x)
 assert torch.allclose(x.square().sum(-1),y.square().sum(-1),atol=1e-6)
def test_scalar_gate_is_simple_control():
 x=torch.randn(2,3,4,5,2);m=ScalarView(Echo());m.scale.data.fill_(2);assert torch.allclose(m(x),2*x)

def test_givens_views_do_not_change_varnet_dc_weights():
 from run_screen import make_model,tie_from
 base=make_model(106011);viewed=tie_from(base,'mirror')
 assert all(torch.equal(a.dc_weight,b.dc_weight) for a,b in zip(base.cascades,viewed.cascades))
 assert all(not hasattr(block.model,'mask') for block in viewed.cascades)
