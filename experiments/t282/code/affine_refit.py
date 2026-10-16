"""Fixed-rate affine refit baseline, NOT a novel FQC/GPTQ implementation.
Group64; unsigned codes; stored offset/positive scale little-endian FP16.
All objective comparisons use the actual serialized metadata/decoded codes.
Original RTN is an admissible fallback in EVERY group. Input weighting is
only a diagonal second-moment proxy, not the full end-to-end task loss.
"""
from __future__ import annotations
import numpy as np
from binary_codec import encode_uniform, pack_bits, unpack_bits

def encode_refit(x, bits, importance=None, iterations=4, factors=(1.,.95,.90), group=64):
    if x.ndim!=2 or x.shape[1]%group or not np.isfinite(x).all() or not 2<=bits<=8:
        raise ValueError('Requires finite 2D group-aligned matrix and 2..8 bits')
    if importance is None: importance=np.ones(x.shape[1],np.float32)
    importance=np.asarray(importance,np.float32)
    if importance.shape!=(x.shape[1],) or not np.isfinite(importance).all() or np.any(importance<=0):
        raise ValueError('Importance must be positive finite per-input-column vector')
    d,original=encode_uniform(x,bits,group); ng=d['groups']; limit=(1<<bits)-1
    meta=np.frombuffer(original[:d['meta_bytes']],'<f2').astype(np.float32).reshape(ng,2).copy()
    codes=unpack_bits(original[d['meta_bytes']:],ng*group,bits).reshape(ng,group).copy()
    a=np.asarray(x,np.float32).reshape(ng,group); wi=importance.reshape(-1,group)
    before=after=regret=0.; changed=0
    for start in range(0,ng,4096):
        stop=min(start+4096,ng); aa=a[start:stop]
        w=wi[np.arange(start,stop)%len(wi)].astype(np.float64)
        sw=w.sum(1); sx=(w*aa).sum(1)
        bestm=meta[start:stop].copy(); bestq=codes[start:stop].copy()
        def error(q,m):
            residual=aa.astype(np.float64)-(q.astype(np.float32)*m[:,1,None]+m[:,0,None]).astype(np.float64)
            return (w*residual*residual).sum(1)
        initial=error(bestq,bestm); besterr=initial.copy()
        for factor in factors:
            m=meta[start:stop].copy()
            if factor!=1.:
                center=m[:,0]+.5*limit*m[:,1]
                scale=np.maximum(m[:,1]*factor,2**-24).astype('<f2').astype(np.float32)
                m=np.column_stack(((center-.5*limit*scale).astype('<f2').astype(np.float32),scale))
            for iteration in range(iterations+1):
                q=np.clip(np.rint((aa-m[:,0,None])/m[:,1,None]),0,limit).astype(np.uint8)
                err=error(q,m); take=err<besterr
                besterr[take]=err[take]; bestm[take]=m[take]; bestq[take]=q[take]
                if iteration==iterations: break
                qq=q.astype(np.float64); sq=(w*qq).sum(1); sqq=(w*qq*qq).sum(1); sxq=(w*aa*qq).sum(1)
                det=sw*sqq-sq*sq; valid=det>np.finfo(np.float64).eps*np.maximum(sw*sqq,1.)
                scale=m[:,1].astype(np.float64).copy()
                scale[valid]=(sw[valid]*sxq[valid]-sx[valid]*sq[valid])/det[valid]
                scale=np.clip(scale,2**-24,65504).astype('<f2').astype(np.float32)
                offset=((sx-scale*sq)/sw).astype('<f2').astype(np.float32)
                m=np.column_stack((offset,scale))
                if not np.isfinite(m).all(): raise FloatingPointError('Invalid metadata')
        before+=float(initial.sum()); after+=float(besterr.sum())
        regret=max(regret,float((besterr-initial).max())); changed+=int((besterr<initial).sum())
        meta[start:stop]=bestm; codes[start:stop]=bestq
    payload=meta.astype('<f2').tobytes()+pack_bits(codes,bits)
    if len(payload)!=len(original) or regret>0: raise AssertionError('rate/fallback property failed')
    return d,payload,{'weighted_SSE_before':before,'weighted_SSE_after':after,
                     'max_group_SSE_increase':regret,'groups_improved':changed,'groups':ng,
                     'payload_bytes':len(payload),'rate_equal_to_RTN':True}
