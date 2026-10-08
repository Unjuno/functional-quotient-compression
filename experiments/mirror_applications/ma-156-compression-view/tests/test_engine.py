import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from engine import D,build,deserialize,make_world,pack_int4,quant4

def test_int4_pack_roundtrip_and_accounting():
 w=torch.tensor([-7,-3,-1,0,1,3,7],dtype=torch.float32);q,s,_=quant4(w);raw=pack_int4(q)
 assert len(raw)==4 and len(raw)*2>=q.numel()

def test_mirror_decode_reconstructs_aligned_views_and_serializes():
 target,base,_,_=make_world(156,True);decoded,payload=build('int4_mirror',target,base)
 assert decoded.shape==(4,D,D) and len(payload)>100
 assert torch.isfinite(decoded).all()
 assert torch.equal(decoded,deserialize(payload))

def test_every_serialized_format_roundtrips():
 target,base,_,_=make_world(157,True)
 for method in ['int4_independent','int4_tied','int4_mirror','int4_shared_rank2','int4_qer_rank2','fp32_untied']:
  decoded,payload=build(method,target,base)
  assert torch.equal(decoded,deserialize(payload)),method
