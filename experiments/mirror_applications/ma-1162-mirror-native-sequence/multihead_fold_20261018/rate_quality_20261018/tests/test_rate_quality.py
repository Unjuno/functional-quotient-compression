"""No second physical weight bank; codes are compact, bit packed, replayable."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import pytest
import torch
from rate_quality import (SharedMultihead,serialize,deserialize,
                          pack_codes,unpack_codes,HEADER_BYTES,shared_bytes,cache_bytes,
                          native_full_bytes,native_folded_qv_bytes)

@pytest.mark.parametrize('code_bits',[8,12,16,32])
def test_angle_codec(code_bits):
    z=[-0.7,0.4,0.8,1.02,-0.1,0.2,-0.3,0.6]
    blob=pack_codes(z,code_bits)
    restored=unpack_codes(blob,len(z),code_bits)
    assert len(blob)==(len(z)*code_bits+7)//8
    tol=1e-6 if code_bits==32 else 2*3.142/(2**code_bits-1)
    assert max(abs(a-b) for a,b in zip(z,restored))<tol

@pytest.mark.parametrize('physical_bits',[16,32])
@pytest.mark.parametrize('code_bits',[8,12,16,32])
def test_roundtrip_payload(physical_bits,code_bits):
    torch.manual_seed(25)
    model=SharedMultihead(16,8,4,'linear').eval()
    with torch.no_grad():
        model.q_code.copy_(torch.tensor([-.62,-.2,.32,.91]))
        model.v_code.copy_(torch.tensor([.67,-.49,.45,-.5]))
    data=serialize(model,physical_bits,code_bits)
    got=deserialize(data)
    assert len(data)==shared_bytes(16,8,physical_bits)+(8*code_bits+7)//8
    assert got.heads==model.heads
    assert sum(p.numel() for p in got.parameters())==8
    x=torch.randn(2,6,16)
    with torch.no_grad():
        diff=(model(x)-got(x)).abs().max().item()
    assert diff < (.06 if code_bits==8 else .012)

def test_non_linear_codec_has_exact_same_module():
    model=SharedMultihead(16,8,4,'nonlinear')
    got=deserialize(serialize(model,32,16))
    assert got.mode=='nonlinear'
    torch.manual_seed(84)
    x=torch.randn(2,6,16)
    with torch.no_grad():
        assert (model(x)-got(x)).abs().max().item()<5e-4

def test_storage_invariant_and_native_comparison():
    assert cache_bytes(8192,4,8,16)==(262144,1048576)
    assert native_full_bytes(16,8,4,16)==14+3072
    assert native_folded_qv_bytes(16,8,4,16)==14+2304
    assert shared_bytes(16,8,16)==14+768
    assert shared_bytes(16,8,16)+8<native_folded_qv_bytes(16,8,4,16)

def test_corruption_rejected():
    model=SharedMultihead(16,8,4,'linear')
    data=serialize(model,16,8)
    for broken in (data[1:],data+b'\x00',b'N'+data[1:]):
        with pytest.raises((ValueError,AssertionError)):
            deserialize(broken)
