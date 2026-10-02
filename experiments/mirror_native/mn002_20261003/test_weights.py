# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
import hashlib
import struct
import pytest
import torch
from model import TinyLM
from weights import save_model,load_model

@pytest.mark.parametrize('kind,hidden,states', [('mirror',32,1),('mirror',32,4),('mirror',32,16),('dense',36,1),('direct_gate',24,1),('full_moe',32,4)])
def test_full_numerical_state_roundtrip_and_size(tmp_path,kind,hidden,states):
    torch.manual_seed(7)
    m=TinyLM(kind=kind,hidden=hidden,states=states,vocab=49,max_length=32).eval()
    p=tmp_path/'m.mnw';save_model(m,p);r=load_model(p)
    assert p.stat().st_size==96+4*sum(v.numel() for v in m.parameters())
    assert m.config==r.config
    for k,v in m.state_dict().items(): assert torch.equal(v,r.state_dict()[k])
    with torch.no_grad():
        x=torch.randint(0,49,(3,12));assert torch.equal(m(x)[0],r(x)[0])

def test_equal_parameter_means_equal_artifact_bytes(tmp_path):
    a=TinyLM(kind='mirror',hidden=32,states=4,vocab=49,max_length=32)
    b=TinyLM(kind='dense',hidden=36,states=1,vocab=49,max_length=32)
    pa=tmp_path/'a.mnw';pb=tmp_path/'b.mnw';save_model(a,pa);save_model(b,pb)
    assert pa.stat().st_size==pb.stat().st_size==70980

@pytest.mark.parametrize('change',['flip','truncate','append','wrong_version_valid_hash','huge_width_valid_hash','nonfinite_valid_hash'])
def test_malformed_artifacts_rejected(tmp_path,change):
    p=tmp_path/'a.mnw';save_model(TinyLM(),p);raw=bytearray(p.read_bytes())
    if change=='flip':raw[75]^=1
    if change=='truncate':raw=raw[:-1]
    if change=='append':raw+=b'x'
    if change=='wrong_version_valid_hash':struct.pack_into('<I',raw,8,999)
    if change=='huge_width_valid_hash':struct.pack_into('<I',raw,16,2**30)
    if change=='nonfinite_valid_hash':struct.pack_into('<f',raw,64,float('nan'))
    if change.endswith('valid_hash'):raw[-32:]=hashlib.sha256(raw[:-32]).digest()
    p.write_bytes(raw)
    with pytest.raises(ValueError):load_model(p)
