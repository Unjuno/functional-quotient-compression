from pathlib import Path
import sys,copy,hashlib,json,struct
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from affine_refit import encode_refit
from binary_codec import encode_uniform,decode_uniform,pack_bits,unpack_bits,write_model,fp16
from independent_decoder import decode_model
from artifact_gate import checked_decode,expected_shapes

@pytest.mark.parametrize('bits',[2,3,4,5,6,7,8])
@pytest.mark.parametrize('case',['normal','weighted','zero','constant','tiny'])
def test_fixed_rate_monotone_and_deterministic(bits,case):
    rng=np.random.default_rng(278); x=rng.normal(0,.2,(3,128)).astype(np.float32);w=np.ones(128,np.float32)
    if case=='weighted':w=np.exp(rng.uniform(-5,5,128)).astype(np.float32)
    if case=='zero':x[:]=0
    if case=='constant':x[:]=.12345
    if case=='tiny':x*=1e-8
    d0,b0=encode_uniform(x,bits);d,b,s=encode_refit(x,bits,w);d2,b2,s2=encode_refit(x,bits,w)
    assert len(b)==len(b0) and b==b2 and s==s2
    a=decode_uniform(d,b);r=decode_uniform(d0,b0)
    loss=((x.astype(np.float64)-a)**2*w).reshape(-1,64).sum(1)
    old=((x.astype(np.float64)-r)**2*w).reshape(-1,64).sum(1)
    assert np.all(loss<=old+1e-20)
    assert np.isfinite(a).all()

@pytest.mark.parametrize('bits',[1,2,3,4,5,6,7,8])
@pytest.mark.parametrize('size',[0,1,63,64,65,262149])
def test_packing_chunks(bits,size):
    q=np.random.default_rng(7).integers(0,1<<bits,size,dtype=np.uint8)
    assert np.array_equal(q,unpack_bits(pack_bits(q,bits),size,bits))

def test_invalid_importance():
    with pytest.raises(ValueError):encode_refit(np.ones((2,64),np.float32),4,np.zeros(64))

@pytest.fixture(scope='module')
def artifact(tmp_path_factory):
    tmp=tmp_path_factory.mktemp('fixture');p=tmp/'valid.fqc'
    cfg={'hidden_size':64,'num_layers':8,'num_heads':16,'vocab_size':12,'max_position_embeddings':16,
         'attention_layers':['global','local']*4,'activation_function':'gelu_new','layer_norm_epsilon':1e-5,'window_size':256}
    sections=[]
    for n,shape in expected_shapes(cfg).items():
        a=np.zeros(shape,np.float32); sections.append(({'kind':'fp16','name':n,'shape':list(shape)},fp16(a).tobytes()))
    write_model(p,cfg,sections,{'synthetic_fixture_only':True})
    return p

def mutate(src,dst,operation):
    blob=src.read_bytes();magic,nh,nb,pad=struct.unpack_from('<8sIQQ',blob)
    h=json.loads(blob[28:28+nh]);body=blob[28+nh:28+nh+nb];operation(h)
    header=json.dumps(h,sort_keys=True,separators=(',',':')).encode()
    out=struct.pack('<8sIQQ',magic,len(header),len(body),0)+header+body
    dst.write_bytes(out+hashlib.sha256(out).digest())

def test_semantic_valid(artifact):assert len(checked_decode(artifact)[1])==108

@pytest.mark.parametrize('case',['rename','layers','hidden','vocab','shape','activation','heads','epsilon'])
def test_semantic_mismatch_with_valid_checksum(artifact,tmp_path,case):
    def edit(h):
        if case=='rename':h['sections'][0]['name']='unknown.weight'
        elif case=='layers':h['config']['num_layers']=7
        elif case=='hidden':h['config']['hidden_size']=128
        elif case=='vocab':h['config']['vocab_size']=13
        elif case=='shape':h['sections'][0]['shape']=list(reversed(h['sections'][0]['shape']))
        elif case=='activation':h['config']['activation_function']='relu'
        elif case=='heads':h['config']['num_heads']=3
        elif case=='epsilon':h['config']['layer_norm_epsilon']=-1
    p=tmp_path/(case+'.fqc');mutate(artifact,p,edit)
    # The historic reader accepts these semantic errors despite the checksum.
    assert len(decode_model(p)[1])==108
    with pytest.raises(ValueError):checked_decode(p)

@pytest.mark.parametrize('case',['truncate','extra','flip','badmagic'])
def test_byte_corruption(artifact,tmp_path,case):
    b=bytearray(artifact.read_bytes())
    if case=='truncate':b=b[:-7]
    elif case=='extra':b+=b'X'
    elif case=='flip':b[150]^=1
    else:b[0]^=1;b[-32:]=hashlib.sha256(b[:-32]).digest()
    p=tmp_path/(case+'.fqc');p.write_bytes(b)
    with pytest.raises(ValueError):checked_decode(p)
