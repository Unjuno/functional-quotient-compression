import io,json,sys,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from model import METHODS,FactorizedExpertDepth,rotate,factorized_input
from engine import make_world,target

def test_givens_and_factorized_input_preserve_norm():
 x=torch.randn(6,16);a=torch.randn(4);b=torch.randn(4);y=factorized_input(x,torch.zeros(6,dtype=torch.long),torch.zeros(6,dtype=torch.long),a[None].expand(4,-1),b[None].expand(4,-1));assert torch.allclose(x.norm(dim=-1),y.norm(dim=-1),atol=1e-5)
def test_method_shapes_and_payloads():
 for method in METHODS:
  m=FactorizedExpertDepth(method,25100);x=torch.randn(7,16);e=torch.randint(4,(7,));l=torch.randint(4,(7,));assert m(x,e,l).shape==(7,12);assert m.serialized_payload_bytes()>0
def test_teacher_modes():
 for mode in ('factorized_axes','independent_pairs'):
  w=make_world(25100,mode);y=target(torch.randn(9,16),torch.randint(4,(9,)),torch.randint(4,(9,)),w,mode);assert y.shape==(9,12);assert torch.isfinite(y).all()
def test_roundtrip_preserves_factorized_logits():
 m=FactorizedExpertDepth('factorized_mirror',25109);raw=m.serialize();cfg={'method':m.method,'address_seed':m.address_seed,'experts':m.experts,'depth':m.depth,'d':m.d,'out':m.out,'rank':m.rank,'factorization':'expert rotates first 8 dims; depth rotates last 8 dims'};meta=json.dumps(cfg,sort_keys=True).encode();obj=torch.load(io.BytesIO(raw[:-len(meta)]),weights_only=False);n=FactorizedExpertDepth(cfg['method'],cfg['address_seed'],cfg['experts'],cfg['depth'],cfg['d'],cfg['out'],cfg['rank']);n.load_state_dict(obj['state_dict']);x=torch.randn(5,16);e=torch.randint(4,(5,));l=torch.randint(4,(5,));assert torch.equal(m(x,e,l),n(x,e,l));assert raw==m.serialize()
