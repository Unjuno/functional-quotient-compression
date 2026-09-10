from pathlib import Path
import sys,struct,hashlib,json,zlib,lzma
import pytest
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'code'))
from lossless_envelope import encode,decode,HEADER,MAGIC,LIMIT

@pytest.mark.parametrize('codec',['identity','zlib9','xz6'])
@pytest.mark.parametrize('data',[b'X',bytes(range(256)),b'\0'*100000, b'reproducibility'*1234])
def test_lossless_roundtrip(codec,data):
    blob=encode(data,codec);assert decode(blob)==data;assert encode(data,codec)==blob

@pytest.mark.parametrize('codec',['identity','zlib9','xz6'])
@pytest.mark.parametrize('case',['truncate','trailing','magic','reserved','codec','oversized','shortraw','longraw','rawhash','outerhash'])
def test_forged_envelope(codec,case):
    blob=bytearray(encode(b'fixture'*4096,codec))
    if case=='truncate':blob=blob[:-1]
    elif case=='trailing':blob+=b'X'
    elif case=='magic':blob[0]^=1
    elif case=='reserved':blob[9]=1
    elif case=='codec':blob[8]=255
    elif case=='oversized':struct.pack_into('<Q',blob,16,LIMIT+1)
    elif case=='shortraw':struct.pack_into('<Q',blob,16,1)
    elif case=='longraw':struct.pack_into('<Q',blob,16,999999)
    elif case=='rawhash':blob[32]^=1
    else:blob[-1]^=1
    if case not in ['truncate','trailing','outerhash']:blob[-32:]=hashlib.sha256(blob[:-32]).digest()
    with pytest.raises(ValueError):decode(bytes(blob))

@pytest.mark.parametrize('codec',['zlib9','xz6'])
@pytest.mark.parametrize('extra',['junk','second_stream'])
def test_trailing_compressed_stream(codec,extra):
    raw=b'fixture'*100;b=encode(raw,codec);payload=b[HEADER.size:-32]
    payload+=b'garbage' if extra=='junk' else (zlib.compress(b'second') if codec=='zlib9' else lzma.compress(b'second'))
    h=HEADER.pack(MAGIC,1 if codec=='zlib9' else 2,len(raw),len(payload),hashlib.sha256(raw).digest());body=h+payload
    with pytest.raises(ValueError):decode(body+hashlib.sha256(body).digest())

@pytest.mark.parametrize('codec',['identity','zlib9','xz6'])
def test_caller_size_limit(codec):
    with pytest.raises(ValueError):decode(encode(b'x'*100,codec),max_raw_bytes=20)

@pytest.mark.parametrize('data',[b'',None,'text'])
def test_invalid_input(data):
    with pytest.raises(ValueError):encode(data)

