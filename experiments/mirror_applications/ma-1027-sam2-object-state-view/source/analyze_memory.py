"""Offline memory reconstruction diagnostic against native per-object SAM2 state."""
import json,pathlib
import numpy as np,torch
E=pathlib.Path(__file__).resolve().parents[1];R=E/'runs'
def read(method):
 raw=(R/f'payload_{method}.bin').read_bytes();h,b=raw.split(b'\n',1);meta=json.loads(h);out={};off=0
 for x in meta['tensors']:
  n=x['bytes'];a=np.frombuffer(b[off:off+n],dtype=np.dtype(x['dtype'])).copy().reshape(x['shape']);out[x['name']]=a;off+=n
 assert off==len(b);return out
def bf(a):return torch.from_numpy(a.astype('uint16',copy=False)).view(torch.bfloat16).float().numpy()
ref=read('independent');nframes=5;methods={}
for method in ['hard_shared','mirror_givens','additive_rank1']:
 cur=read(method);allerr=[];per=[]
 for f in range(nframes):
  target=bf(ref[f'memory/frame_{f}/raw'])
  base=bf(cur[f'memory/frame_{f}/base'])
  if method=='hard_shared':rec=np.repeat(base[None],len(target),axis=0)
  elif method=='mirror_givens':
   angles=cur[f'memory/frame_{f}/angles'];rec=[]
   for a in angles:
    x=base.copy()
    for k,(i,j) in enumerate(((0,1),(2,3))):
     c,s=np.cos(float(a[k])),np.sin(float(a[k]));u=x[i].copy();v=x[j].copy();x[i]=c*u-s*v;x[j]=s*u+c*v
    rec.append(x)
   rec=np.stack(rec)
  else:
   basis=bf(cur[f'memory/frame_{f}/basis']);coef=cur[f'memory/frame_{f}/coeff'];rec=base[None]+coef[:,None,None,None]*basis[None]
  e=np.sqrt(np.mean((rec-target)**2))/max(np.sqrt(np.mean(target**2)),1e-12);per.append(float(e));allerr.append((rec-target).reshape(-1))
 methods[method]={'mean_relative_rmse':float(np.mean(per)),'per_frame_relative_rmse':per}
# Record fit diagnostics separately from segmentation quality.
out={'reference':'uncompressed per-object SAM2 maskmem_features saved in independent payload','metrics':methods};(R/'memory_reconstruction.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
