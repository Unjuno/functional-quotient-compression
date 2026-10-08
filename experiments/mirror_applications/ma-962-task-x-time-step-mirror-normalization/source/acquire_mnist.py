"""Download only MNIST training IDX files before the audit gate."""
from __future__ import annotations
import gzip, hashlib, json, struct, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'source' / 'data'
BASE = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
FILES = {'train_images':'train-images-idx3-ubyte.gz','train_labels':'train-labels-idx1-ubyte.gz'}

def fetch(name: str, filename: str) -> bytes:
    path=DATA/filename
    if not path.exists():
        req=urllib.request.Request(BASE+filename,headers={'User-Agent':'MA-962-research/1.0'})
        with urllib.request.urlopen(req,timeout=60) as resp: path.write_bytes(resp.read())
    return path.read_bytes()

def acquire_train() -> dict:
    DATA.mkdir(parents=True,exist_ok=True); arrays={}; files={}
    for key,filename in FILES.items():
        compressed=fetch(key,filename); raw=gzip.decompress(compressed)
        files[filename]={'compressed_bytes':len(compressed),'compressed_sha256':hashlib.sha256(compressed).hexdigest(),'decoded_bytes':len(raw)}
        if key=='train_images':
            magic,n,h,w=struct.unpack('>IIII',raw[:16]); assert (magic,n,h,w)==(2051,60000,28,28)
            arrays[key]=torch_frombuffer(raw[16:],(n,784))
        else:
            magic,n=struct.unpack('>II',raw[:8]); assert (magic,n)==(2049,60000)
            arrays[key]=torch_frombuffer(raw[8:],(n,))
    manifest={'source_url':BASE,'files':files,'train_split':[0,50000],'development_split':[50000,60000],
              'audit_test_files_downloaded':False,'audit_test_files_decoded':False}
    import torch
    torch.save(arrays['train_images'],DATA/'train_images.pt'); torch.save(arrays['train_labels'],DATA/'train_labels.pt')
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

def torch_frombuffer(raw: bytes, shape: tuple[int,...]):
    import torch
    return torch.frombuffer(bytearray(raw),dtype=torch.uint8).clone().reshape(shape)

def acquire_audit() -> tuple:
    """Call only after frozen development gates pass."""
    import torch
    DATA.mkdir(parents=True,exist_ok=True)
    vals=[]; files={}
    for key,filename in (('test_images','t10k-images-idx3-ubyte.gz'),('test_labels','t10k-labels-idx1-ubyte.gz')):
        compressed=fetch(key,filename); raw=gzip.decompress(compressed)
        files[filename]={'compressed_bytes':len(compressed),'compressed_sha256':hashlib.sha256(compressed).hexdigest(),'decoded_bytes':len(raw)}
        if key=='test_images':
            magic,n,h,w=struct.unpack('>IIII',raw[:16]); assert (magic,n,h,w)==(2051,10000,28,28)
            vals.append(torch_frombuffer(raw[16:],(n,784)))
        else:
            magic,n=struct.unpack('>II',raw[:8]); assert (magic,n)==(2049,10000)
            vals.append(torch_frombuffer(raw[8:],(n,)))
    manifest=json.loads((DATA/'manifest.json').read_text()); manifest['test_files']=files; manifest['audit_test_files_downloaded']=True; manifest['audit_test_files_decoded']=True
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return tuple(vals)

if __name__=='__main__': print(json.dumps(acquire_train(),indent=2))
