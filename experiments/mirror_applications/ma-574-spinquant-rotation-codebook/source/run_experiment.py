#!/usr/bin/env python3
"""MA-574 Pythia multi-layer int4 rotation-codebook screen."""
from __future__ import annotations
import argparse,hashlib,json,struct,time
from pathlib import Path
import numpy as np

MODEL_SHA='3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd'
MODEL_REV='e93a9faa9c77e5d09219f6c868bfc7a1bd65593c'
LAYERS=tuple(range(6));WIDTH=512;GROUP=32;BLOCKS=WIDTH//GROUP;K=4;NCAND=16


def stream_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read_tensor(path,name):
    with open(path,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0];head=json.loads(f.read(n));x=head[name]
        if x['dtype']!='F16':raise ValueError((name,x['dtype']))
        a,b=x['data_offsets'];f.seek(8+n+a);raw=f.read(b-a)
    return np.frombuffer(raw,dtype='<f2').reshape(x['shape']).astype(np.float32)


def fwht(a):
    x=np.array(a,dtype=np.float32,copy=True);n=x.shape[-1];h=1
    while h<n:
        v=x.reshape(*x.shape[:-1],n//(2*h),2,h);left=v[...,0,:].copy();right=v[...,1,:].copy()
        v[...,0,:]=left+right;v[...,1,:]=left-right;h*=2
    return x


def new_code(rng):
    signs=rng.choice(np.array([-1,1],np.int8),size=(BLOCKS,GROUP))
    perms=np.stack([rng.permutation(GROUP).astype(np.uint8) for _ in range(BLOCKS)])
    return signs,perms


def view_weight(w,code):
    signs,perms=code;x=fwht(w.reshape(w.shape[0],BLOCKS,GROUP))*(1/np.sqrt(GROUP));x*=signs[None,:,:]
    return np.take_along_axis(x,np.broadcast_to(perms[None,:,:],x.shape),axis=-1).reshape(w.shape)


def quant_int4(w):
    g=w.reshape(w.shape[0],BLOCKS,GROUP);s=(np.max(np.abs(g),axis=-1)/7.).astype(np.float16)
    s=np.maximum(s.astype(np.float32),1e-8).astype(np.float16);den=s.astype(np.float32)[...,None]
    q=np.clip(np.rint(g/den),-7,7).astype(np.int8);dq=q.astype(np.float32)*den
    u=(q+8).astype(np.uint8).reshape(w.shape);packed=(u[:,0::2]<<4)|u[:,1::2]
    return packed.astype(np.uint8),s,dq.reshape(w.shape)


def unpack_int4(packed,scales):
    h=(packed>>4).astype(np.int8)-8;l=(packed&15).astype(np.int8)-8
    q=np.stack((h,l),axis=-1).reshape(packed.shape[0],WIDTH).astype(np.float32)
    return (q.reshape(packed.shape[0],BLOCKS,GROUP)*scales.astype(np.float32)[...,None]).reshape(packed.shape[0],WIDTH)


def row_split(n,seed):
    r=np.random.default_rng(seed).permutation(n);c=np.sort(r[:int(.2*n)]);a=np.sort(r[int(.2*n):]);return c,a


def relative_error(a,b,idx):
    x=a[idx].astype(np.float64);y=b[idx].astype(np.float64)
    return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-12))


def modules_from_checkpoint(model_path):
    items=[]
    for layer in LAYERS:
        for short,key in [('attn',f'gpt_neox.layers.{layer}.attention.dense.weight'),('mlp_up',f'gpt_neox.layers.{layer}.mlp.dense_h_to_4h.weight')]:
            items.append({'id':f'L{layer}_{short}','layer':layer,'name':key,'w':read_tensor(model_path,key)})
    return items


def choose_codebook(err_cal,train_idx,k=K):
    chosen=[];remaining=set(range(err_cal.shape[1]))
    for _ in range(k):
        scores=[]
        for j in sorted(remaining):
            candidate=chosen+[j];score=float(np.mean(np.min(err_cal[np.asarray(train_idx)][:,candidate],axis=1)))
            scores.append((score,j))
        j=min(scores)[1];chosen.append(j);remaining.remove(j)
    return np.asarray(chosen,dtype=np.int16)


def pack_method(out,method,seed,items,packed_by,scales_by,codes,ids,metadata=None):
    qcat=np.concatenate([packed_by[i].ravel() for i in range(len(items))])
    scat=np.concatenate([scales_by[i].ravel() for i in range(len(items))])
    offsets=np.zeros((len(items),2),np.int64);offq=offs=0
    for i in range(len(items)):
        offsets[i]=[offq,offs];offq+=packed_by[i].size;offs+=scales_by[i].size
    qpath=out/f'{method}_weights.npz'
    arrays={'qweight_int4':qcat,'group_scales_fp16':scat,'module_offsets':offsets,
      'module_shapes':np.asarray([x['w'].shape for x in items],np.int32),
      'module_names':np.asarray([x['id'].encode() for x in items],dtype='S16'),
      'seed':np.array([seed],np.int32),'model_sha256':np.frombuffer(MODEL_SHA.encode(),dtype='S64'),
      'schema':np.array([574,1],np.int32)}
    np.savez(qpath,**arrays);qbytes=qpath.stat().st_size;cbytes=0;cpath=None
    if codes is not None:
        cpath=out/f'{method}_rotation_codes.npz'
        ca={'signs':np.stack([codes[j][0] for j in range(len(codes))]),'permutations':np.stack([codes[j][1] for j in range(len(codes))]),
            'module_code_ids':np.asarray(ids,dtype=np.uint8),'seed':np.array([seed],np.int32),'schema':np.array([574,2],np.int32)}
        if metadata:ca.update(metadata)
        np.savez(cpath,**ca);cbytes=cpath.stat().st_size
    elif metadata:
        cpath=out/f'{method}_view_scales.npz';np.savez(cpath,seed=np.array([seed],np.int32),schema=np.array([574,3],np.int32),**metadata);cbytes=cpath.stat().st_size
    return {'total_payload_bytes':qbytes+cbytes,'weight_payload_bytes':qbytes,'rotation_code_bytes':cbytes,'weights_file':qpath.name,'codes_file':None if cpath is None else cpath.name}


def run(seed,split,model_dir,out):
    out.mkdir(parents=True,exist_ok=True);model_path=model_dir/'model.safetensors'
    if stream_sha(model_path)!=MODEL_SHA:raise ValueError('pinned model hash mismatch')
    t0=time.perf_counter();items=modules_from_checkpoint(model_path);load_s=time.perf_counter()-t0
    for i,item in enumerate(items):item['cal'],item['audit']=row_split(item['w'].shape[0],57400+i)
    train=[i for i,x in enumerate(items) if x['layer']<=3];held=[i for i,x in enumerate(items) if x['layer']>=4]
    rng=np.random.default_rng(seed);codes=[new_code(rng) for _ in range(NCAND)]
    errors_cal=np.zeros((len(items),NCAND));errors_audit=np.zeros_like(errors_cal);encode_s=np.zeros(NCAND)
    for j,code in enumerate(codes):
        t0=time.perf_counter()
        for i,x in enumerate(items):
            vw=view_weight(x['w'],code);_,_,dq=quant_int4(vw)
            errors_cal[i,j]=relative_error(dq,vw,x['cal']);errors_audit[i,j]=relative_error(dq,vw,x['audit'])
        encode_s[j]=time.perf_counter()-t0
    t0=time.perf_counter();cb=choose_codebook(errors_cal,train);global_id=min(range(NCAND),key=lambda j:float(np.mean(errors_cal[train,j])));cbfit_s=time.perf_counter()-t0
    random_cb=np.arange(K,dtype=np.int16)
    assignments={
      'identity_int4':np.full(len(items),-1,np.int16),
      'global_quarot':np.full(len(items),global_id,np.int16),
      'independent16':np.argmin(errors_cal,axis=1).astype(np.int16),
      'shared_codebook4':np.asarray([min(cb,key=lambda j:errors_cal[i,j]) for i in range(len(items))],np.int16),
      'random_codebook4':np.asarray([min(random_cb,key=lambda j:errors_cal[i,j]) for i in range(len(items))],np.int16),
      'native_codebook4':np.asarray([min(cb,key=lambda j:errors_cal[i,j]) for i in range(len(items))],np.int16)}
    use_codes={'global_quarot':[global_id],'independent16':list(assignments['independent16']),
      'shared_codebook4':list(cb),'random_codebook4':list(random_cb),'native_codebook4':list(cb)}
    method_files={};method_metrics={};full_delta=0.
    # Identity and SmoothQuant do not carry orthogonal codes.
    for method in ('identity_int4','global_quarot','independent16','shared_codebook4','random_codebook4','native_codebook4','smoothquant'):
        t0=time.perf_counter();packed_by=[];scales_by=[];cal_err=[];audit_err=[];extra_scales=[]
        for i,x in enumerate(items):
            if method=='identity_int4':
                trans=x['w'];back=lambda z:z;codeidx=-1
            elif method=='smoothquant':
                colmax=np.max(np.abs(x['w'][x['cal']]),axis=0).astype(np.float64);center=np.exp(np.mean(np.log(np.maximum(colmax,1e-12))))
                scale=np.clip(np.sqrt(np.maximum(colmax,1e-12)/center),.25,4.).astype(np.float16).astype(np.float32)
                trans=x['w']*scale[None,:];back=lambda z,s=scale:z/s[None,:];extra_scales.append(scale.astype(np.float16));codeidx=-2
            else:
                codeidx=int(assignments[method][i]);trans=view_weight(x['w'],codes[codeidx]);back=lambda z:z;
                # For an orthogonal coordinate change the Frobenius reconstruction error is invariant to inverse rotation.
            p,s,dq=quant_int4(trans);dq_actual=unpack_int4(p,s)
            if method=='smoothquant':
                metric_pred=back(dq_actual);metric_target=x['w']
            elif method=='identity_int4':
                metric_pred=dq_actual;metric_target=x['w']
            else:
                metric_pred=dq_actual;metric_target=trans
            packed_by.append(p);scales_by.append(s)
            cal_err.append(relative_error(metric_pred,metric_target,x['cal']))
            audit_err.append(relative_error(metric_pred,metric_target,x['audit']))
            if codeidx>=0:
                xx=np.random.default_rng(seed+500+i).normal(size=(2,WIDTH)).astype(np.float32)
                qx=view_input(xx,codes[codeidx]);fw=view_weight(x['w'][:32],codes[codeidx]);full_delta=max(full_delta,float(np.max(np.abs(xx@x['w'][:32].T-qx@fw.T))))
        if method=='identity_int4':codes_arg=None;ids_arg=None;meta=None
        elif method=='smoothquant':codes_arg=None;ids_arg=None;meta={'channel_scale_fp16':np.stack(extra_scales)}
        elif method=='global_quarot':codes_arg=[codes[global_id]];ids_arg=np.zeros(len(items),np.uint8);meta={'global_code_id':np.array([global_id],np.uint8)}
        elif method=='independent16':codes_arg=[codes[int(k)] for k in assignments[method]];ids_arg=np.arange(len(items),dtype=np.uint8);meta=None
        elif method=='random_codebook4':codes_arg=[codes[int(k)] for k in random_cb];ids_arg=np.asarray([list(random_cb).index(int(k)) for k in assignments[method]],np.uint8);meta={'candidate_ids':random_cb.astype(np.uint8)}
        else:codes_arg=[codes[int(k)] for k in cb];ids_arg=np.asarray([list(cb).index(int(k)) for k in assignments[method]],np.uint8);meta={'candidate_ids':cb.astype(np.uint8)}
        sizes=pack_method(out,method,seed,items,packed_by,scales_by,codes_arg,ids_arg,meta);method_files[method]=sizes
        method_metrics[method]={'total_payload_bytes':sizes['total_payload_bytes'],'weight_payload_bytes':sizes['weight_payload_bytes'],'rotation_code_bytes':sizes['rotation_code_bytes'],
          'mean_train_layer_nrmse':float(np.mean([cal_err[i] for i in train])),'mean_heldout_layer_nrmse':float(np.mean([audit_err[i] for i in held])),
          'max_heldout_matrix_nrmse':float(max(audit_err[i] for i in held)),'per_module_calibration_nrmse':cal_err,'per_module_heldout_nrmse':audit_err,'encode_and_package_seconds':time.perf_counter()-t0}
    # Oracle independent best-of-16 heldout assignment upper reference from the same candidates.
    independent_held=float(np.mean([min(errors_audit[i,:]) for i in held]))
    alias=method_metrics['shared_codebook4']['per_module_heldout_nrmse']==method_metrics['native_codebook4']['per_module_heldout_nrmse'] and method_files['shared_codebook4']['rotation_code_bytes']==method_files['native_codebook4']['rotation_code_bytes']
    rep={'experiment_id':'MA-574','seed':seed,'split':split,'model_revision':MODEL_REV,'model_sha256':MODEL_SHA,
      'matrices':[{'id':x['id'],'layer':x['layer'],'shape':list(x['w'].shape),'weight_sha256':hashlib.sha256(x['w'].tobytes()).hexdigest()} for x in items],
      'train_layers':[0,1,2,3],'heldout_layers':[4,5],'calibration_fraction':.2,'candidate_count':NCAND,'codebook_candidate_ids':cb.tolist(),'global_code_id':int(global_id),
      'independent_best_heldout_nrmse':independent_held,'full_precision_function_max_abs_error':full_delta,'methods':method_metrics,'file_sizes':method_files,
      'compute':{'model_load_seconds':load_s,'candidate_view_encode_seconds_total':float(encode_s.sum()),'codebook_fit_seconds':cbfit_s,'view_ops_per_module_token':BLOCKS*GROUP*int(np.log2(GROUP))+WIDTH,'view_ops_per_all_12_module_tokens':12*(BLOCKS*GROUP*int(np.log2(GROUP))+WIDTH),'optimizer_updates':0},
      'native_codebook_alias':{'metrics_and_selected_rotation_state_exact':alias}}
    (out/'metrics.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n')
    (out/'splits.json').write_text(json.dumps({'train_module_ids':[items[i]['id'] for i in train],'heldout_module_ids':[items[i]['id'] for i in held],'candidate_seed':seed,'codebook_candidate_ids':cb.tolist()},indent=2)+'\n')
    print(json.dumps({'seed':seed,'codebook':cb.tolist(),'independent_heldout_nrmse':independent_held,'methods':{k:{a:b for a,b in v.items() if not a.startswith('per_module')} for k,v in method_metrics.items()},'native_alias':alias},indent=2))


def view_input(x,code):
    signs,perms=code;z=fwht(x.reshape(*x.shape[:-1],BLOCKS,GROUP))*(1/np.sqrt(GROUP));z*=signs
    return np.take_along_axis(z,np.broadcast_to(perms,z.shape),axis=-1).reshape(x.shape)


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--split',choices=['dev','fresh'],required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.seed,a.split,a.model_dir,a.out)

if __name__=='__main__':main()
