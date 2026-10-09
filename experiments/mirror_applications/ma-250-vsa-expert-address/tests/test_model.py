import io,json,sys,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from model import METHODS,ExpertViews,address_matrices
from engine import make_world,target

def test_binding_operators_are_orthogonal():
 for method in ('map','hadamard','hrr'):
  mats=address_matrices(25000,method)
  eye=torch.eye(16).expand(4,-1,-1)
  assert torch.allclose(mats.transpose(1,2)@mats,eye,atol=1e-5)
def test_method_shapes_and_serialization():
 for method in METHODS:
  m=ExpertViews(method,25000);x=torch.randn(9,16);role=torch.randint(4,(9,))
  assert m(x,role).shape==(9,12);assert m.serialized_payload_bytes()>0
def test_teacher_modes_produce_regression_targets():
 for mode in ('aligned_shared_base','independent_roles'):
  w=make_world(25000,mode);y=target(torch.randn(7,16),torch.randint(4,(7,)),w,mode);assert y.shape==(7,12);assert torch.isfinite(y).all()
def test_serialization_roundtrip_reconstructs_fixed_codes():
 m=ExpertViews('hadamard',25007);x=torch.randn(5,16);role=torch.randint(4,(5,));raw=m.serialize();cfg={'method':m.method,'address_seed':m.address_seed,'roles':m.roles,'d':m.d,'out':m.out,'rank':m.rank,'address_algorithm':'map-sign-permutation / hadamard-HdiagH / hrr-unit-spectrum-v1'};meta=json.dumps(cfg,sort_keys=True).encode();obj=torch.load(io.BytesIO(raw[:-len(meta)]),weights_only=False);clone=ExpertViews(cfg['method'],cfg['address_seed'],cfg['roles'],cfg['d'],cfg['out'],cfg['rank']);clone.load_state_dict(obj['state_dict']);assert torch.equal(m(x,role),clone(x,role));assert raw==m.serialize()
