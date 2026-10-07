import io,json,sys,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from model import METHODS,TopKExperts,givens
from engine import make_world,role_of,target

def test_givens_preserve_norm_and_role_labels():
 x=torch.randn(11,16);a=torch.randn(4,8);assert torch.allclose(x.norm(dim=-1),givens(x,a[torch.arange(11)%4]).norm(dim=-1),atol=1e-5);assert torch.equal(role_of(x),((x[:,0]>0).long()*2+(x[:,1]>0).long()) )
def test_method_outputs_and_payloads():
 for method in METHODS:
  m=TopKExperts(method,30000);y,logits,r=m(torch.randn(9,16));assert y.shape==(9,12) and logits.shape==(9,4) and r.shape==(9,);assert m.serialized_payload_bytes()>0
def test_teacher_modes_and_router_classes():
 x=torch.tensor([[1.,1.],[-1.,1.],[1.,-1.],[-1.,-1.]]+[[0.]*2]*0)
 # Pad each input to the declared hidden width.
 x=torch.cat((x,torch.zeros(4,14)),dim=-1);roles=role_of(x);assert roles.tolist()==[3,1,2,0]
 for mode in ('aligned_views','independent_experts'):
  w=make_world(30000,mode);assert target(torch.randn(8,16),torch.randint(4,(8,)),w,mode).shape==(8,12)
def test_serialization_roundtrip():
 m=TopKExperts('mirror',30009);x=torch.randn(5,16);raw=m.serialize();cfg={'method':m.method,'address_seed':m.address_seed,'experts':m.experts,'d':m.d,'out':m.out,'rank':m.rank,'router_rank':m.router_rank,'top_k':1};meta=json.dumps(cfg,sort_keys=True).encode();obj=torch.load(io.BytesIO(raw[:-len(meta)]),weights_only=False);n=TopKExperts(cfg['method'],cfg['address_seed'],cfg['experts'],cfg['d'],cfg['out'],cfg['rank'],cfg['router_rank']);n.load_state_dict(obj['state_dict']);assert torch.equal(m(x)[0],n(x)[0]);assert raw==m.serialize()
