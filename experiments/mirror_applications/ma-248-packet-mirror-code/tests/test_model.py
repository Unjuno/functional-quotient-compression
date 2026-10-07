import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import torch
from model import PacketDecoder
from engine import make_world,sample

def test_all_methods_shapes_and_serialization():
    for mode,codes in [('correlated_two_packet_codes',2),('independent_sixteen_packet_codes',16)]:
        table=make_world(24800);batch=sample(table,5,mode,7)
        for method in PacketDecoder.METHODS:
            m=PacketDecoder(method,4,codes)
            assert m(*batch[:4]).shape==(5,4,16)
            assert m.serialized_payload_bytes()>0

def test_givens_views_preserve_norm():
    m=PacketDecoder('mirror',4,16);z=torch.randn(9,4,16);pairs=z.reshape(9,4,8,2);ang=m.angles[None];c,s=torch.cos(ang),torch.sin(ang)
    out=torch.stack((c*pairs[...,0]-s*pairs[...,1],s*pairs[...,0]+c*pairs[...,1]),-1)
    assert torch.allclose(z.square().sum(-1),out.square().sum((-1,-2)),atol=1e-5)

def test_sample_contains_valid_teacher_trajectories():
    table=make_world(24801)
    for mode in ('correlated_two_packet_codes','independent_sixteen_packet_codes'):
        rule,start,bits,code,target=sample(table,32,mode,123);prev=start
        for j in range(4):
            assert torch.equal(table[bits[:,j],rule,prev],target[:,j]);prev=target[:,j]
