"""Bounded, checksummed lossless FQC envelope (stdlib only).
Format: <8sB7xQQ32s> header, compressed payload, SHA256(header+payload).
All sizes include the 96-byte header/checksum overhead. Not a new quantizer.
"""
from __future__ import annotations
import hashlib,lzma,struct,zlib
from pathlib import Path
MAGIC=b'FQZ00001';HEADER=struct.Struct('<8sB7xQQ32s');LIMIT=512*1024*1024
CODECS={'identity':0,'zlib9':1,'xz6':2}

def encode(raw:bytes,codec:str='xz6')->bytes:
    if not isinstance(raw,bytes) or not raw or len(raw)>LIMIT:raise ValueError('input type/extent')
    if codec not in CODECS:raise ValueError('unknown codec')
    payload=raw if codec=='identity' else zlib.compress(raw,9) if codec=='zlib9' else lzma.compress(raw,format=lzma.FORMAT_XZ,check=lzma.CHECK_CRC64,preset=6)
    h=HEADER.pack(MAGIC,CODECS[codec],len(raw),len(payload),hashlib.sha256(raw).digest());body=h+payload
    return body+hashlib.sha256(body).digest()

def decode(blob:bytes,max_raw_bytes:int=LIMIT)->bytes:
    if not isinstance(blob,bytes) or len(blob)<HEADER.size+32:raise ValueError('short/invalid envelope')
    if not 0<max_raw_bytes<=LIMIT:raise ValueError('invalid caller limit')
    if hashlib.sha256(blob[:-32]).digest()!=blob[-32:]:raise ValueError('envelope checksum')
    magic,codec,nraw,ncompressed,expected=HEADER.unpack_from(blob)
    if magic!=MAGIC or codec not in CODECS.values() or any(blob[9:16]):raise ValueError('schema/reserved')
    if not 0<nraw<=max_raw_bytes or ncompressed>LIMIT+1048576 or len(blob)!=HEADER.size+ncompressed+32:raise ValueError('extent')
    payload=blob[HEADER.size:-32]
    try:
        if codec==0:raw=payload
        elif codec==1:
            decoder=zlib.decompressobj();raw=decoder.decompress(payload,nraw+1)
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:raise ValueError('zlib incomplete/trailing stream')
        else:
            decoder=lzma.LZMADecompressor(format=lzma.FORMAT_XZ,memlimit=256*1024*1024);raw=decoder.decompress(payload,max_length=nraw+1)
            if not decoder.eof or decoder.unused_data:raise ValueError('xz incomplete/trailing stream')
    except (lzma.LZMAError,zlib.error) as e:raise ValueError('compressed stream') from e
    if len(raw)!=nraw or hashlib.sha256(raw).digest()!=expected:raise ValueError('restored length/hash')
    return raw

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['pack','unpack']);p.add_argument('source',type=Path);p.add_argument('target',type=Path);p.add_argument('--codec',choices=list(CODECS),default='xz6');a=p.parse_args()
    if a.target.exists():raise FileExistsError(a.target)
    data=a.source.read_bytes();out=encode(data,a.codec) if a.operation=='pack' else decode(data)
    a.target.parent.mkdir(parents=True,exist_ok=True)
    with a.target.open('xb') as f:f.write(out)
    print(a.target,len(out),hashlib.sha256(out).hexdigest())
if __name__=='__main__':main()
