#!/usr/bin/env python3
"""MA-576 SmoothQuant + residual rotation int4 matrix screen."""
from __future__ import annotations
import argparse, hashlib, json, struct, time
from pathlib import Path
import numpy as np
MODEL_SHA='3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd'
MODEL_REV='e93a9faa9c77e5d09219f6c868bfc7a1bd65593c'
WIDTH=512; GROUP=32; BLOCKS=16; K=16; LAYERS=range(6)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read_tensor(path,name):
 with open(path,'rb') as f:
  n=struct.unpack('<Q',f.read(8))[0]; head=json.loads(f.read(n)); x=head[name]
  if x['dtype']!='F16':raise ValueError((name,x['dtype']))
  a,b=x['data_offsets'];f.seek(8+n+a);raw=f.read(b-a)
 return np.frombuffer(raw,dtype='<f2').reshape(x['shape']).astype(np.float32)
def fwht(a):
 x=np.array(a,dtype=np.float32,copy=True); n=x.shape[-1]; h=1
 while h<n:
  v=x.reshape(*x.shape[:-1],n//(2*h),2,h);l=v[...,0,:].copy();r=v[...,1,:].copy();v[...,0,:]=l+r;v[...,1,:]=l-r;h*=2
 return x
def new_code(rng):
 return rng.choice(np.array([-1,1],np.int8),size=GROUP),rng.permutation(GROUP).astype(np.uint8)
def basis(code):
 s,p=code;eye=np.eye(GROUP,dtype=np.float32);h=fwht(eye)*(1/np.sqrt(GROUP));h*=s[None,:]
 return h[:,p]
def rotate(w,code):
 x=fwht(w.reshape(w.shape[0],BLOCKS,GROUP))*(1/np.sqrt(GROUP));s,p=code
 return (x*s[None,None,:])[:,:,p].reshape(w.shape)
def unrotate(wq,code):
 x=wq.reshape(wq.shape[0],BLOCKS,GROUP);q=basis(code)
 return np.einsum('mbi,ji->mbj',x,q).reshape(wq.shape)
def row_split(n,seed):
 r=np.random.default_rng(seed).permutation(n);k=int(.2*n);return np.sort(r[:k]),np.sort(r[k:])
def quant(w):
 g=w.reshape(w.shape[0],BLOCKS,GROUP);s=(np.max(np.abs(g),axis=-1)/7.).astype(np.float16);s=np.maximum(s.astype(np.float32),1e-8).astype(np.float16)
 q=np.clip(np.rint(g/s.astype(np.float32)[...,None]),-7,7).astype(np.int8);u=(q+8).astype(np.uint8).reshape(w.shape)
 return ((u[:,0::2]<<4)|u[:,1::2]).astype(np.uint8),s
def dequant(p,s):
 h=(p>>4).astype(np.int8)-8;l=(p&15).astype(np.int8)-8;q=np.stack((h,l),-1).reshape(p.shape[0],WIDTH).astype(np.float32)
 return (q.reshape(p.shape[0],BLOCKS,GROUP)*s.astype(np.float32)[...,None]).reshape(p.shape[0],WIDTH)
def error(a,b,rows):
 x=a[rows].astype(np.float64);y=b[rows].astype(np.float64);return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-12))
def smooth_scale(w,cal):
 c=np.max(np.abs(w[cal]),axis=0).astype(np.float64);g=np.exp(np.mean(np.log(np.maximum(c,1e-12))));return np.clip(np.sqrt(np.maximum(c,1e-12)/g),.25,4.).astype(np.float16).astype(np.float32)
def reconstruct(w,code,smooth):
 t=w*(smooth[None,:] if smooth is not None else 1.)
 if code is not None:t=rotate(t,code)
 p,sc=quant(t);d=dequant(p,sc)
 if code is not None:d=unrotate(d,code)
 if smooth is not None:d=d/smooth[None,:]
 return p,sc,d
def pack(out,method,seed,items,quantized,channel_scales,codes,ids):
 qcat=np.concatenate([p.ravel() for p,s in quantized]);scat=np.concatenate([s.ravel() for p,s in quantized]);wpath=out/f'{method}_weights.npz'
 np.savez(wpath,qweight_int4=qcat,group_scales_fp16=scat,module_shapes=np.asarray([x['w'].shape for x in items],np.int32),module_names=np.asarray([x['id'].encode() for x in items],dtype='S16'),seed=np.array([seed],np.int32),model_sha256=np.frombuffer(MODEL_SHA.encode(),dtype='S64'),schema=np.array([576,1],np.int32))
 files=[wpath];sizes={'weight_payload_bytes':wpath.stat().st_size,'rotation_code_bytes':0,'channel_scale_bytes':0}
 if channel_scales is not None:
  p=out/f'{method}_channel_scales.npz';np.savez(p,channel_scale_fp16=np.stack(channel_scales).astype(np.float16),seed=np.array([seed],np.int32),schema=np.array([576,2],np.int32));files.append(p);sizes['channel_scale_bytes']=p.stat().st_size
 if codes is not None:
  p=out/f'{method}_rotation_codes.npz';np.savez(p,signs=np.stack([c[0] for c in codes]),permutations=np.stack([c[1] for c in codes]),module_code_ids=np.asarray(ids,np.uint8),seed=np.array([seed],np.int32),schema=np.array([576,3],np.int32));files.append(p);sizes['rotation_code_bytes']=p.stat().st_size
 sizes.update(total_payload_bytes=sum(p.stat().st_size for p in files),files=[p.name for p in files]);return sizes
def run(seed,split,model_dir,out):
 wall=time.perf_counter();model=model_dir/'model.safetensors'
 if sha(model)!=MODEL_SHA:raise ValueError('pinned model hash mismatch')
 t=time.perf_counter();items=[]
 for layer in LAYERS:
  for short,key in [('attn',f'gpt_neox.layers.{layer}.attention.dense.weight'),('mlp_up',f'gpt_neox.layers.{layer}.mlp.dense_h_to_4h.weight')]:
   w=read_tensor(model,key);cal,audit=row_split(w.shape[0],57600+len(items));items.append({'id':f'L{layer}_{short}','layer':layer,'w':w,'cal':cal,'audit':audit})
 model_load=time.perf_counter()-t;out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(seed);codes=[new_code(rng) for _ in range(K)]
 scales=[smooth_scale(x['w'],x['cal']) for x in items]
 t=time.perf_counter();calerr=np.empty((len(items),K));auditerr=np.empty_like(calerr);code_times=[]
 for i,x in enumerate(items):
  for j,c in enumerate(codes):
   q,ss,d=reconstruct(x['w'],c,scales[i]);calerr[i,j]=error(d,x['w'],x['cal']);auditerr[i,j]=error(d,x['w'],x['audit'])
 candidate_seconds=time.perf_counter()-t;train=list(range(len(items)));held=list(range(len(items)))
 ids=np.argmin(calerr,axis=1).astype(np.uint8);global_id=min(range(K),key=lambda j:float(np.mean(calerr[train,j])));random_ids=np.zeros(len(items),np.uint8)
 methods={};files={}
 config={
  'identity_int4':(None,None,None),
  'smoothquant_only':(None,scales,None),
  'quarot_global':(np.full(len(items),global_id,np.uint8),None,None),
  'smoothquant_quarot_global':(np.full(len(items),global_id,np.uint8),scales,None),
  'smoothquant_residual_mirror':(ids,scales,codes),
  'smoothquant_random_residual':(random_ids,scales,codes),
  'native_smoothquant_quarot':(ids,scales,codes)}
 native_payload={}
 for method,(mid,ss,cbank) in config.items():
  t=time.perf_counter();qs=[];ce=[];ae=[]
  for i,x in enumerate(items):
   if method=='identity_int4':smooth=None;code=None
   elif method=='smoothquant_only':smooth=ss[i];code=None
   elif method=='quarot_global':smooth=None;code=cbank[int(mid[i])] if cbank else codes[global_id]
   elif method=='smoothquant_quarot_global':smooth=ss[i];code=codes[global_id]
   elif method=='smoothquant_random_residual':smooth=ss[i];code=codes[0]
   else:smooth=ss[i];code=codes[int(mid[i])]
   p,sc,d=reconstruct(x['w'],code,smooth);qs.append((p,sc));ce.append(error(d,x['w'],x['cal']));ae.append(error(d,x['w'],x['audit']))
  method_codes=None;method_ids=None;ch=None
  if method in ('quarot_global','smoothquant_quarot_global'):
   code=codes[global_id];method_codes=[code];method_ids=np.zeros(len(items),np.uint8)
  elif method in ('smoothquant_residual_mirror','native_smoothquant_quarot'):
   method_codes=[codes[int(j)] for j in ids];method_ids=np.arange(len(items),dtype=np.uint8)
  elif method=='smoothquant_random_residual':
   method_codes=[codes[0]];method_ids=np.zeros(len(items),np.uint8)
  elif method=='smoothquant_only':ch=ss
  if method in ('smoothquant_residual_mirror','native_smoothquant_quarot','smoothquant_random_residual','smoothquant_quarot_global'):ch=ss
  sizes=pack(out,method,seed,items,qs,ch,method_codes,method_ids);files[method]=sizes
  methods[method]={'total_payload_bytes':sizes['total_payload_bytes'],'weight_payload_bytes':sizes['weight_payload_bytes'],'rotation_code_bytes':sizes['rotation_code_bytes'],'channel_scale_bytes':sizes['channel_scale_bytes'],'mean_calibration_nrmse':float(np.mean([ce[i] for i in train])),'mean_audit_nrmse':float(np.mean([ae[i] for i in held])),'max_heldout_matrix_nrmse':float(max(ae[i] for i in held)),'encode_seconds':time.perf_counter()-t}
 alias=(methods['smoothquant_residual_mirror']['mean_audit_nrmse']==methods['native_smoothquant_quarot']['mean_audit_nrmse'] and files['smoothquant_residual_mirror']['total_payload_bytes']==files['native_smoothquant_quarot']['total_payload_bytes'])
 rep={'experiment_id':'MA-576','seed':seed,'split':split,'model_revision':MODEL_REV,'model_sha256':MODEL_SHA,'evaluation':'calibration/audit output-row split per matrix; no held-out layer claim','candidate_count':K,'global_code_id':global_id,'per_matrix_code_ids':ids.tolist(),'methods':methods,'file_sizes':files,'native_alias':alias,'compute':{'model_load_seconds':model_load,'candidate_calibration_seconds':candidate_seconds,'per_token_view_ops_per_module':BLOCKS*GROUP*5+WIDTH,'optimizer_updates':0,'total_wall_seconds':time.perf_counter()-wall}}
 (out/'metrics.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n');print(json.dumps({'seed':seed,'global_code':global_id,'ids':ids.tolist(),'methods':{k:{q:v for q,v in x.items() if q not in ('encode_seconds',)} for k,x in methods.items()},'native_alias':alias},indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--split',choices=['dev','fresh'],required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.seed,a.split,a.model_dir,a.out)
if __name__=='__main__':main()
