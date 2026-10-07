import sys,io,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from model import HeadStudent,givens
from engine import world,targets

def test_givens_preserves_norm():
 x=torch.randn(5,4,16);theta=torch.randn(4,8);y=givens(x,theta);assert torch.allclose(x.norm(dim=-1),y.norm(dim=-1),atol=1e-5)
def test_method_shapes_and_payloads():
 q,_=torch.linalg.qr(torch.randn(16,16))
 for method in ('mtp','tied','scalar_gate','lowrank','mirror'):
  m=HeadStudent(method,q);assert m(torch.randn(7,16)).shape==(7,4,12);assert m.serialized_payload_bytes()>0
def test_teacher_modes_have_valid_distributions():
 for mode in ('aligned_shared_base','independent_offset_heads'):
  w=world(24900,mode);p=targets(torch.randn(11,16),w,mode);assert p.shape==(11,4,12);assert torch.allclose(p.sum(-1),torch.ones(11,4),atol=1e-6)

def test_payload_roundtrip_preserves_exact_logits():
 import json
 m=HeadStudent('mirror',torch.eye(16));x=torch.randn(9,16);raw=m.serialize();cfg={'method':m.method,'offsets':m.offsets,'vocab':m.vocab,'d':m.d,'rank':m.rank};meta=json.dumps(cfg,sort_keys=True).encode()
 obj=torch.load(io.BytesIO(raw[:-len(meta)]),weights_only=False);clone=HeadStudent(cfg['method'],torch.eye(16),cfg['offsets'],cfg['vocab'],cfg['d'],cfg['rank']);clone.load_state_dict(obj['state_dict'])
 assert torch.equal(m(x),clone(x));assert raw==m.serialize() and len(raw)==m.serialized_payload_bytes()
