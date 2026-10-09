#!/usr/bin/env python3
"""MA-575 factorized layer x input-block rotation code screen."""
from __future__ import annotations
import argparse, hashlib, json, struct, time
from pathlib import Path
import numpy as np

MODEL_SHA='3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd'
MODEL_REV='e93a9faa9c77e5d09219f6c868bfc7a1bd65593c'
WIDTH=512; GROUP=32; BLOCKS=16; K=16; LAYERS=tuple(range(6))

def read_tensor(path,name):
    with open(path,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]; head=json.loads(f.read(n)); x=head[name]
        if x['dtype']!='F16': raise ValueError((name,x['dtype']))
        a,b=x['data_offsets']; f.seek(8+n+a); raw=f.read(b-a)
    return np.frombuffer(raw,dtype='<f2').reshape(x['shape']).astype(np.float32)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for z in iter(lambda:f.read(8*1024*1024),b''): h.update(z)
    return h.hexdigest()

def fwht(a):
    x=np.array(a,dtype=np.float32,copy=True); n=x.shape[-1]; h=1
    while h<n:
        v=x.reshape(*x.shape[:-1],n//(2*h),2,h); l=v[...,0,:].copy(); r=v[...,1,:].copy()
        v[...,0,:]=l+r; v[...,1,:]=l-r; h*=2
    return x

def new_code(rng):
    return rng.choice(np.array([-1,1],np.int8),GROUP).astype(np.int8),rng.permutation(GROUP).astype(np.uint8)

def compose_codes(layer,block):
    sl,pl=layer; sb,pb=block
    return (sb.astype(np.int8)*sl[pb].astype(np.int8)).astype(np.int8),pl[pb].astype(np.uint8)

def row_split(n,seed):
    r=np.random.default_rng(seed).permutation(n); c=np.sort(r[:int(.2*n)]); a=np.sort(r[int(.2*n):]); return c,a

def quant_int4(w):
    g=w.reshape(w.shape[0],BLOCKS,GROUP); s=(np.max(np.abs(g),axis=-1)/7.).astype(np.float16)
    s=np.maximum(s.astype(np.float32),1e-8).astype(np.float16); den=s.astype(np.float32)[...,None]
    q=np.clip(np.rint(g/den),-7,7).astype(np.int8); u=(q+8).astype(np.uint8).reshape(w.shape)
    return (u[:,0::2]<<4)|u[:,1::2],s

def quant_group(w):
    scale=(np.max(np.abs(w),axis=-1)/7.).astype(np.float16); scale=np.maximum(scale.astype(np.float32),1e-8).astype(np.float16)
    q=np.clip(np.rint(w/scale.astype(np.float32)[:,None]),-7,7).astype(np.int8); d=q.astype(np.float32)*scale.astype(np.float32)[:,None]
    return d

def unpack(p,s):
    h=(p>>4).astype(np.int8)-8; l=(p&15).astype(np.int8)-8
    q=np.stack((h,l),axis=-1).reshape(p.shape[0],WIDTH).astype(np.float32)
    return (q.reshape(p.shape[0],BLOCKS,GROUP)*s.astype(np.float32)[...,None]).reshape(p.shape)

def transform(w,codes):
    x=fwht(w.reshape(w.shape[0],BLOCKS,GROUP))*(1/np.sqrt(GROUP))
    out=np.empty_like(x)
    for b,c in enumerate(codes):
        s,p=c; out[:,b,:]=x[:,b,:][:,p]*s[None,:]
    return out.reshape(w.shape)

def err(pred,target,rows):
    a=pred[rows].astype(np.float64); b=target[rows].astype(np.float64)
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12))

def payload(out,method,seed,items,weights,codes,ids,meta=None):
    qcat=np.concatenate([x[0].ravel() for x in weights]); scat=np.concatenate([x[1].ravel() for x in weights])
    wpath=out/f'{method}_weights.npz'
    np.savez(wpath,qweight_int4=qcat,group_scales_fp16=scat,module_shapes=np.asarray([x['w'].shape for x in items],np.int32),module_names=np.asarray([x['id'].encode() for x in items],dtype='S16'),seed=np.array([seed],np.int32),model_sha256=np.frombuffer(MODEL_SHA.encode(),dtype='S64'),schema=np.array([575,1],np.int32))
    cbytes=0; cfile=None
    if codes is not None:
      cpath=out/f'{method}_rotation_codes.npz'; arrays={'signs':np.stack([c[0] for c in codes]),'permutations':np.stack([c[1] for c in codes]),'ids':np.asarray(ids,dtype=np.uint8),'seed':np.array([seed],np.int32),'schema':np.array([575,2],np.int32)}
      if meta: arrays.update(meta)
      np.savez(cpath,**arrays); cbytes=cpath.stat().st_size; cfile=cpath.name
    return {'total_payload_bytes':wpath.stat().st_size+cbytes,'weight_payload_bytes':wpath.stat().st_size,'rotation_code_bytes':cbytes,'weights_file':wpath.name,'codes_file':cfile}

def run(seed,split,model_dir,out):
    out.mkdir(parents=True,exist_ok=True); mp=model_dir/'model.safetensors'
    if sha(mp)!=MODEL_SHA: raise ValueError('pinned checkpoint hash mismatch')
    items=[]
    for layer in LAYERS:
        for short,key in [('attn',f'gpt_neox.layers.{layer}.attention.dense.weight'),('mlp_up',f'gpt_neox.layers.{layer}.mlp.dense_h_to_4h.weight')]:
            w=read_tensor(mp,key); cal,audit=row_split(w.shape[0],57500+len(items)); items.append({'id':f'L{layer}_{short}','layer':layer,'w':w,'cal':cal,'audit':audit})
    rng=np.random.default_rng(seed); bank=[new_code(rng) for _ in range(K)]
    # Precompute held-out/calibration reconstruction error for every factor pair per matrix and block.
    ec=np.empty((len(items),BLOCKS,K,K),np.float32); ea=np.empty_like(ec)
    for i,it in enumerate(items):
      for b in range(BLOCKS):
       wb=it['w'][:,b*GROUP:(b+1)*GROUP]
       for a in range(K):
        for z in range(K):
         c=compose_codes(bank[a],bank[z]); v=fwht(wb)*(1/np.sqrt(GROUP)); v=v[:,c[1]]*c[0][None,:]
         d=quant_group(v)
         ec[i,b,a,z]=err(d,v,it['cal']); ea[i,b,a,z]=err(d,v,it['audit'])
    train=[i for i,x in enumerate(items) if x['layer']<4]; held=[i for i,x in enumerate(items) if x['layer']>=4]
    # Fit only training layers. Fixed two coordinate-descent sweeps are part of the frozen protocol.
    t0=time.perf_counter(); lid=np.zeros(6,np.uint8); bid=np.zeros(BLOCKS,np.uint8)
    for _ in range(2):
      for layer in range(4):
       ii=[i for i in train if items[i]['layer']==layer]
       lid[layer]=min(range(K),key=lambda a:float(np.mean([ec[i,b,a,int(bid[b])] for i in ii for b in range(BLOCKS)])))
      for b in range(BLOCKS):
       bid[b]=min(range(K),key=lambda z:float(np.mean([ec[i,b,int(lid[items[i]['layer']]),z] for i in train])))
    fit_seconds=time.perf_counter()-t0
    lid[4:]=0
    # Independent upper and layer-shared controls choose from the same fixed candidate factors.
    independent=np.argmin(ec,axis=(2,3)); layer_shared=np.zeros((len(items),BLOCKS),np.uint8)
    for layer in LAYERS:
      ii=[i for i,x in enumerate(items) if x['layer']==layer]
      for b in range(BLOCKS):
       c=min(range(K),key=lambda z:float(np.mean([ec[i,b,z,z] for i in ii])))
       for i in ii: layer_shared[i,b]=c
    random_lid=np.random.default_rng(seed+7).integers(0,K,size=6,dtype=np.uint8); random_bid=np.random.default_rng(seed+9).integers(0,K,size=BLOCKS,dtype=np.uint8)
    # Independent and layer-shared states store selected exact per-block codes; factored state stores only bank+IDs.
    methods={}; state={}
    for method in ('identity_int4','independent_matrix_block','layer_shared','random_factorized','factorized_layer_block','native_boft_factorized'):
      quant=[]; calerr=[]; auditerr=[]; saved_codes=[]; saved_ids=[]
      for i,it in enumerate(items):
       cs=[]
       for b in range(BLOCKS):
        if method=='identity_int4': c=(np.ones(GROUP,np.int8),np.arange(GROUP,dtype=np.uint8))
        elif method=='independent_matrix_block': c=bank[int(independent[i,b])]; saved_codes.append(c); saved_ids.append(int(independent[i,b]))
        elif method=='layer_shared': c=bank[int(layer_shared[i,b])]; saved_codes.append(c); saved_ids.append(int(layer_shared[i,b]))
        elif method=='random_factorized': c=compose_codes(bank[int(random_lid[it['layer']])],bank[int(random_bid[b])])
        else: c=compose_codes(bank[int(lid[it['layer']])],bank[int(bid[b])])
        cs.append(c)
       v=transform(it['w'],cs); p,s=quant_int4(v); d=unpack(p,s); quant.append((p.astype(np.uint8),s))
       if method=='identity_int4': target=it['w']
       else: target=v
       calerr.append(err(d,target,it['cal'])); auditerr.append(err(d,target,it['audit']))
      if method in ('identity_int4',): cstore=None; ids=None; meta=None
      elif method in ('independent_matrix_block','layer_shared'): cstore=saved_codes; ids=saved_ids; meta={'mode':np.array([method.encode()])}
      elif method=='random_factorized': cstore=bank; ids=np.concatenate([random_lid,random_bid]); meta={'layer_ids':random_lid,'block_ids':random_bid,'mode':np.array([b'random-factor'])}
      else: cstore=bank; ids=np.concatenate([lid,bid]); meta={'layer_ids':lid,'block_ids':bid,'mode':np.array([method.encode()])}
      sz=payload(out,method,seed,items,quant,cstore,ids,meta)
      methods[method]={'total_payload_bytes':sz['total_payload_bytes'],'weight_payload_bytes':sz['weight_payload_bytes'],'rotation_code_bytes':sz['rotation_code_bytes'],'train_layer_nrmse':float(np.mean([calerr[i] for i in train])),'heldout_layer_nrmse':float(np.mean([auditerr[i] for i in held])),'max_heldout_matrix_nrmse':float(max(auditerr[i] for i in held)),'encode_seconds':0.0}
      state[method]=sz
    alias=(methods['factorized_layer_block']['heldout_layer_nrmse']==methods['native_boft_factorized']['heldout_layer_nrmse'] and state['factorized_layer_block']['rotation_code_bytes']==state['native_boft_factorized']['rotation_code_bytes'])
    report={'experiment_id':'MA-575','seed':seed,'split':split,'model_revision':MODEL_REV,'model_sha256':MODEL_SHA,'layers_train':[0,1,2,3],'layers_heldout':[4,5],'candidate_count':K,'fitted_layer_ids':lid.tolist(),'fitted_block_ids':bid.tolist(),'methods':methods,'file_sizes':state,'native_factor_alias':alias,'compute':{'factor_fit_seconds':fit_seconds,'view_ops_per_module_token':BLOCKS*GROUP*5+WIDTH,'optimizer_updates':0}}
    (out/'metrics.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps({'seed':seed,'fit_layer_ids':lid.tolist(),'fit_block_ids':bid.tolist(),'metrics':methods,'native_alias':alias},indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--split',choices=['dev','fresh'],required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.seed,a.split,a.model_dir,a.out)
if __name__=='__main__':main()
