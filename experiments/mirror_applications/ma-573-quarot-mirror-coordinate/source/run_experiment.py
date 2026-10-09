#!/usr/bin/env python3
"""MA-573 block-Hadamard symmetry and groupwise int4 coordinate assay."""
from __future__ import annotations
import argparse, hashlib, json, struct, time
from pathlib import Path
import numpy as np

MODEL_SHA='3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd'
MODEL_REV='e93a9faa9c77e5d09219f6c868bfc7a1bd65593c'
V,D,GROUP,CANDIDATES=50304,512,32,16
BLOCKS=D//GROUP


def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read_safetensor(path:Path,name:str):
    with open(path,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n));item=header[name]
        if item['dtype']!='F16':raise ValueError(f"expected F16 checkpoint tensor, got {item['dtype']}")
        start,end=item['data_offsets'];f.seek(8+n+start);raw=f.read(end-start)
    return np.frombuffer(raw,dtype='<f2').reshape(item['shape']).astype(np.float32)


def fwht(a):
    """Unnormalized orthogonal-family Hadamard butterfly along the last axis."""
    x=np.array(a,dtype=np.float32,copy=True);n=x.shape[-1];h=1
    while h<n:
        view=x.reshape(*x.shape[:-1],n//(2*h),2,h)
        left=view[...,0,:].copy();right=view[...,1,:].copy()
        view[...,0,:]=left+right;view[...,1,:]=left-right;h*=2
    return x


def sample_code(rng):
    signs=rng.choice(np.array([-1,1],np.int8),size=(BLOCKS,GROUP))
    perms=np.stack([rng.permutation(GROUP).astype(np.uint8) for _ in range(BLOCKS)])
    return signs,perms


def apply_weight_view(w,signs,perms):
    x=fwht(w.reshape(w.shape[0],BLOCKS,GROUP))*(1/np.sqrt(GROUP))
    x*=signs[None,:,:]
    return np.take_along_axis(x,np.broadcast_to(perms[None,:,:],x.shape),axis=-1).reshape(w.shape)


def apply_input_view(x,signs,perms):
    z=fwht(x.reshape(*x.shape[:-1],BLOCKS,GROUP))*(1/np.sqrt(GROUP))
    z*=signs
    return np.take_along_axis(z,np.broadcast_to(perms,z.shape),axis=-1).reshape(x.shape)


def quantize_int4(matrix):
    groups=matrix.reshape(matrix.shape[0],BLOCKS,GROUP)
    scales=np.max(np.abs(groups),axis=-1)/7.0
    scales=np.maximum(scales,1e-8).astype(np.float16)
    decoded_scales=scales.astype(np.float32)[...,None]
    q=np.clip(np.rint(groups/decoded_scales),-7,7).astype(np.int8)
    decoded=q.astype(np.float32)*decoded_scales
    biased=(q+8).astype(np.uint8).reshape(matrix.shape)
    packed=((biased[:,0::2]<<4)|biased[:,1::2]).astype(np.uint8)
    return packed,scales,decoded.reshape(matrix.shape)


def dequantize_int4(packed,scales):
    hi=(packed>>4).astype(np.int8)-8;lo=(packed&15).astype(np.int8)-8
    q=np.stack((hi,lo),axis=-1).reshape(packed.shape[0],-1).astype(np.float32)
    return (q.reshape(packed.shape[0],BLOCKS,GROUP)*scales.astype(np.float32)[...,None]).reshape(packed.shape[0],-1)


def relative_error(pred,target,rows):
    a=pred[rows].astype(np.float64);b=target[rows].astype(np.float64)
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12))


def write_payload(path,packed,scales,method,seed,extra=None):
    arrays={'qweight_int4_packed':packed,'group_scales_fp16':scales,
            'method':np.frombuffer(method.encode(),dtype=f'S{len(method)}'),
            'seed':np.array([seed],np.int32),'model_sha256':np.frombuffer(MODEL_SHA.encode(),dtype='S64'),
            'model_revision':np.frombuffer(MODEL_REV.encode(),dtype='S40'),
            'group_size':np.array([GROUP],np.int16),'schema':np.array([573,1],np.int32)}
    if extra:arrays.update(extra)
    np.savez(path,**arrays);return path.stat().st_size


def run(seed,split,model_dir,out):
    out.mkdir(parents=True,exist_ok=True);model_path=model_dir/'model.safetensors'
    if file_sha(model_path)!=MODEL_SHA:raise ValueError('pinned model checksum mismatch')
    t0=time.perf_counter();w=read_safetensor(model_path,'embed_out.weight')
    if w.shape!=(V,D):raise ValueError(f'unexpected W shape {w.shape}')
    load_s=time.perf_counter()-t0
    split_rng=np.random.default_rng(57300);row_ids=split_rng.permutation(V);ncal=int(.2*V);cal_rows=np.sort(row_ids[:ncal]);audit_rows=np.sort(row_ids[ncal:])
    w_sha=hashlib.sha256(w.tobytes()).hexdigest()
    rng=np.random.default_rng(seed);codes=[sample_code(rng) for _ in range(CANDIDATES)]
    cand=[];enc_s=[]
    for ci,(signs,perms) in enumerate(codes):
        t0=time.perf_counter();wp=apply_weight_view(w,signs,perms);packed,scales,dq=quantize_int4(wp)
        cal=relative_error(dq,wp,cal_rows);aud=relative_error(dq,wp,audit_rows)
        cand.append({'index':ci,'calibration_nrmse':cal,'heldout_nrmse':aud,'packed':packed,'scales':scales,'decoded':dq,'view':wp})
        enc_s.append(time.perf_counter()-t0)
    best=min(cand,key=lambda x:x['calibration_nrmse']);fixed=cand[0]
    # SmoothQuant diagonal view from calibration-row column maxima, with paid FP16 scales.
    t0=time.perf_counter();colmax=np.max(np.abs(w[cal_rows]),axis=0).astype(np.float64)
    center=np.exp(np.mean(np.log(np.maximum(colmax,1e-12))))
    smooth=np.clip(np.sqrt(np.maximum(colmax,1e-12)/center),.25,4.).astype(np.float16).astype(np.float32)
    ws=w*smooth[None,:];sp,ss,sdq=quantize_int4(ws);s_eff=sdq/smooth[None,:]
    smooth_cal=relative_error(s_eff,w,cal_rows);smooth_audit=relative_error(s_eff,w,audit_rows);smooth_s=time.perf_counter()-t0
    p0,sc0,dq0=quantize_int4(w)
    id_cal=relative_error(dq0,w,cal_rows);id_audit=relative_error(dq0,w,audit_rows)
    # Exact full-precision symmetry check on a held-out batch and selected sweep code.
    x=np.random.default_rng(seed+9901).normal(size=(8,D)).astype(np.float32)
    xq=apply_input_view(x,*codes[best['index']]);wf=best['view']
    before=x@w[:64].T;after=xq@wf[:64].T;exact_delta=float(np.max(np.abs(before-after)))
    common_extra={'calibration_rows':np.array([ncal],np.int32),'audit_rows':np.array([len(audit_rows)],np.int32)}
    baseline_bytes=write_payload(out/'identity_int4.npz',p0,quantize_int4(w)[1],'identity',seed,common_extra)
    fixed_bytes=write_payload(out/'quarot_single.npz',fixed['packed'],fixed['scales'],'quarot_single',seed,
      {'signs':codes[0][0],'permutations':codes[0][1],**common_extra})
    mirror_bytes=write_payload(out/'mirror_sweep.npz',best['packed'],best['scales'],'mirror_sweep',seed,
      {'signs':codes[best['index']][0],'permutations':codes[best['index']][1],**common_extra})
    native_bytes=write_payload(out/'native_quarot_sweep.npz',best['packed'],best['scales'],'native_quarot_sweep',seed,
      {'signs':codes[best['index']][0],'permutations':codes[best['index']][1],**common_extra})
    smooth_bytes=write_payload(out/'smoothquant.npz',sp,ss,'smoothquant',seed,{'channel_scale_fp16':smooth.astype(np.float16),**common_extra})
    qflops=2*V*D;view_ops=BLOCKS*GROUP*int(np.log2(GROUP))+D
    with np.load(out/'mirror_sweep.npz') as ma,np.load(out/'native_quarot_sweep.npz') as na:
        alias_arrays=all(np.array_equal(ma[k],na[k]) for k in ma.files if k!='method')
    methods={
      'identity_int4':{'payload_bytes':baseline_bytes,'calibration_nrmse':id_cal,'heldout_nrmse':id_audit,'encode_seconds':0.0,'transform_ops_per_token':0},
      'quarot_single':{'payload_bytes':fixed_bytes,'calibration_nrmse':fixed['calibration_nrmse'],'heldout_nrmse':fixed['heldout_nrmse'],'encode_seconds':float(enc_s[0]),'transform_ops_per_token':view_ops},
      'mirror_sweep':{'payload_bytes':mirror_bytes,'calibration_nrmse':best['calibration_nrmse'],'heldout_nrmse':best['heldout_nrmse'],'encode_seconds':float(sum(enc_s)),'transform_ops_per_token':view_ops},
      'native_quarot_sweep':{'payload_bytes':native_bytes,'calibration_nrmse':best['calibration_nrmse'],'heldout_nrmse':best['heldout_nrmse'],'encode_seconds':float(sum(enc_s)),'transform_ops_per_token':view_ops},
      'smoothquant':{'payload_bytes':smooth_bytes,'calibration_nrmse':smooth_cal,'heldout_nrmse':smooth_audit,'encode_seconds':float(smooth_s),'transform_ops_per_token':D}}
    alias_metrics=all(methods['mirror_sweep'][k]==methods['native_quarot_sweep'][k] for k in methods['mirror_sweep'])
    report={'experiment_id':'MA-573','seed':seed,'split':split,'model_revision':MODEL_REV,'model_sha256':MODEL_SHA,'output_weight_sha256':w_sha,'weight_shape':[V,D],
      'calibration_rows':int(ncal),'audit_rows':int(len(audit_rows)),'candidate_count':CANDIDATES,'selected_candidate_index':int(best['index']),
      'exact_full_precision_max_abs_error':exact_delta,'methods':methods,
      'compute':{'model_load_seconds':load_s,'random_view_quantize_total_seconds':float(sum(enc_s)),'smoothquant_encode_seconds':float(smooth_s),'quantized_matvec_flops_per_token':qflops,'view_transform_ops_per_token':view_ops,'view_transform_fraction_of_quantized_matvec':view_ops/qflops,'optimizer_updates':0},
      'native_sweep_alias':{'mirror_and_native_code_equal':True,'payload_arrays_equal':alias_arrays,'metrics_equal':alias_metrics}}
    (out/'metrics.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    (out/'split.json').write_text(json.dumps({'partition_seed':57300,'calibration_row_ids':cal_rows.tolist(),'audit_row_ids':audit_rows.tolist(),'selected_code_seed':seed,'selected_candidate_index':int(best['index'])},indent=2)+'\n')
    print(json.dumps({'seed':seed,'split':split,'selected':int(best['index']),'function_max_abs_error':exact_delta,'methods':methods},indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--split',choices=['dev','fresh'],required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.seed,a.split,a.model_dir,a.out)

if __name__=='__main__':main()
