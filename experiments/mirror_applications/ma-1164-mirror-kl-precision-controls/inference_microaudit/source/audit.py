"""Posthoc inference memory-format audit for MA-1164; no training claims.

Native and Mirror serialize one FP16 contiguous array with an identical 22-byte header.
Mirror also serializes a 1210-bit address bitset, not a tensor of expanded weights.
The existing mirror operator still materializes all virtual weights at forward time:
this experiment does NOT claim a weight-free streaming kernel.
"""
import csv
import io
import json
import math
import struct
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[2] / 'source'
sys.path.insert(0, str(SRC))
from core import CoeffDecoder, N, logits, state_bytes, native_payload

MAGIC_M = b'MRRFQC01'
MAGIC_N = b'NATFQC01'
HEADER = '<8sIIHHH'
HEADER_LEN = struct.calcsize(HEADER)


def mirror_pack(indices, coords, k, rounds):
    idx=np.asarray(indices.cpu().numpy(), dtype=np.int64)
    coord=np.asarray(coords.detach().cpu().numpy(), dtype=np.float16)
    order=np.argsort(idx)
    idx=idx[order]
    coord=coord[order]
    assert len(np.unique(idx))==len(idx) and idx.min()>=0 and idx.max()<N
    bitmap=np.zeros(N,dtype=np.uint8)
    bitmap[idx]=1
    payload=struct.pack(HEADER, MAGIC_M,N,len(idx),k,rounds,16)
    return payload+np.packbits(bitmap,bitorder='little').tobytes()+coord.astype('<f2').tobytes()


def mirror_unpack(payload):
    magic,n,m,k,r,bits=struct.unpack_from(HEADER,payload)
    assert magic==MAGIC_M and n==N and bits==16
    bm_len=(n+7)//8
    assert len(payload)==HEADER_LEN+bm_len+2*m
    mask=np.unpackbits(np.frombuffer(payload,dtype=np.uint8,count=bm_len,offset=HEADER_LEN),bitorder='little')[:n]
    idx=np.flatnonzero(mask)
    assert len(idx)==m
    coords=np.frombuffer(payload,dtype='<f2',count=m,offset=HEADER_LEN+bm_len).copy()
    model=CoeffDecoder(idx,'mirror',k,r).eval()
    with torch.no_grad():model.coords.copy_(torch.from_numpy(coords.astype(np.float64)))
    return model


def native_pack(flat):
    a=np.asarray(flat.detach().cpu().to(torch.float16).numpy(),dtype='<f2')
    return struct.pack(HEADER,MAGIC_N,N,N,0,0,16)+a.tobytes()


def native_unpack(payload):
    magic,n,m,k,r,bits=struct.unpack_from(HEADER,payload)
    assert magic==MAGIC_N and n==m==N and k==r==0 and bits==16
    assert len(payload)==HEADER_LEN+2*N
    arr=np.frombuffer(payload,dtype='<f2',count=N,offset=HEADER_LEN).copy()
    return torch.from_numpy(arr.astype(np.float64))


def median_ms(fn,repeat=29):
    with torch.no_grad():
        for _ in range(5):fn()
        obs=[]
        for _ in range(repeat):
            t=time.perf_counter(); fn(); obs.append((time.perf_counter()-t)*1000)
    return statistics.median(obs),statistics.quantiles(obs,n=10)[-1]


def main():
    torch.set_num_threads(1)
    torch.manual_seed(27)
    indices=torch.randperm(N)[:768].sort().values
    decoder=CoeffDecoder(indices,'mirror',8,4).eval()
    with torch.no_grad():decoder.coords.copy_(torch.randn(768,dtype=torch.float64)*.4)
    packed=mirror_pack(decoder.indices,decoder.coords,decoder.k,decoder.rounds)
    restored=mirror_unpack(packed)
    # Equivalence to the FP16 quantization of the ORIGINAL decoder coefficients
    with torch.no_grad():
        q=decoder.coords.to(torch.float16).double()
        baseline=CoeffDecoder(decoder.indices,'mirror',decoder.k,decoder.rounds).eval()
        baseline.coords.copy_(q)
        assert torch.equal(baseline.weights(),restored.weights())
        x=torch.randn(64,64,dtype=torch.float64)
        assert torch.equal(baseline(x),restored(x))
        w=baseline.weights()
    native=native_pack(w)
    assert torch.equal(native_unpack(native),w.half().double())
    torch_mirror=state_bytes({'kind':'mirror','k':8,'rounds':4,'state':{'indices':indices.to(torch.int16),'coords':decoder.coords.half()}})
    torch_native=state_bytes(native_payload(w,torch.float16))
    np.save(Path(__file__).resolve().parents[1] / 'results' / 'weights.npy',w.detach().cpu().numpy())
    rows=[]
    modes=[('native_F16_flat',len(native),None),('mirror_F16_bitmask',len(packed),None),('native_F16_torch_multi',torch_native,None),('mirror_F16_torch_int16_index',torch_mirror,None)]
    for kind,size,_ in modes:
        rows.append({'method':kind,'persistent_bytes':size,'ratio_to_native_raw':size/len(native)})
    with torch.no_grad():
        qmirror=mirror_unpack(packed)
        nn_median,nn_p90=median_ms(lambda:logits(w,x))
        virtual_median,virtual_p90=median_ms(lambda:qmirror.weights())
        model_median,model_p90=median_ms(lambda:qmirror(x))
        load_median,load_p90=median_ms(lambda:mirror_unpack(packed),repeat=30)
    result={'date':'2026-10-18','evidence':'posthoc deterministic synthetic-parameter serialization and CPU runtime audit; NOT learned digit accuracy',
      'hardware':'CPU PyTorch 2.10.0+cpu one thread, x86 virtual AMD EPYC; CPU clock not fixed',
      'shape':{'virtual_weights':N,'physical_coords':768,'K':8,'R':4,'batch':64,'input_features':64},
      'serialized':{row['method']:row['persistent_bytes'] for row in rows},
      'runtime_median_ms':{'native_matmul_only':nn_median,'virtual_weight_generation_only':virtual_median,
                            'mirror_total_forward':model_median,'bitmask_decode_once':load_median},
      'runtime_p90_ms':{'native_matmul_only':nn_p90,'virtual_weight_generation_only':virtual_p90,
                         'mirror_total_forward':model_p90,'bitmask_decode_once':load_p90},
      'quantized_decoder_roundtrip_exact':True,
      'virtual_weights_stored_persistently':False,
      'virtual_weights_materialized_per_forward':True,
      'mask_bits':N,
      'shared_runtime_code_bytes_excluded_from_all_payloads':True,
      'limits':['synthetic coefficient values; source core.py unmodified; K/R dense virtual vector still materialized',
                'not a streaming output kernel','FP16 is storage; compute is FP64',
                'raw packing relies on stable runtime code version; its deploy bytes must be counted at system level']}
    out=Path(__file__).resolve().parents[1] / 'results'
    out.mkdir(parents=True, exist_ok=True)
    with (out/'results.csv').open('w',newline='') as f:
        d=csv.DictWriter(f,fieldnames=list(rows[0]));d.writeheader();d.writerows(rows)
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (out/'mirror_packed.bin').write_bytes(packed)
    (out/'native_packed.bin').write_bytes(native)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
