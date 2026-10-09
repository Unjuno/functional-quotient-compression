"""MA-301 continuous task-view masks versus packed binary supermasks."""
import argparse,csv,hashlib,json,struct,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];D,N,R=4096,64,2;DEV=[30100,30101];FRESH=[30110,30111,30112];SEEDS=[0,1,2]
torch.set_num_threads(2)
def make(world,seed):
 g=torch.Generator().manual_seed(world*100003+seed*7919+301)
 basis=torch.randn(D,R,generator=g);basis=basis/torch.linalg.vector_norm(basis,dim=0,keepdim=True)
 codes=torch.randn(N,R,generator=g)*2.0
 # Task masks are thresholded continuous shared-basis views.
 logits=codes@basis.T; masks=(logits>0).to(torch.uint8)
 x=torch.randn(N,128,D,generator=g)
 return basis,codes,masks,x
def blob(meta,raw):
 m=json.dumps(meta,sort_keys=True,separators=(',',':')).encode();return b'MA301\0'+struct.pack('<I',len(m))+m+raw
def pack_tensors(method,parts):
 meta={'method':method,'shapes':[list(p.shape) for p in parts],'dtypes':[str(p.dtype) for p in parts]}
 return blob(meta,b''.join(p.detach().cpu().contiguous().numpy().tobytes() for p in parts))
def pack_masks(masks,method):
 raw=np.packbits(masks.detach().cpu().numpy().reshape(-1),bitorder='little').tobytes();return blob({'method':method,'shape':list(masks.shape),'bitorder':'little'},raw)
def run(phase):
 rows=[]
 for world in DEV if phase=='development' else FRESH:
  for seed in SEEDS:
   basis,codes,masks,x=make(world,seed);target=masks.float()
   # Continuous Mirror code: charge fp16 shared logits basis and task codes, then threshold.
   mb=basis.half();mc=codes.half();mirror=((mc.float()@mb.float().T)>0).float()
   # Generic PCA control on binary mask values; mean, factors, and codes are all charged.
   z=target*2-1;mean=z.mean(0);u,s, vh=torch.linalg.svd(z-mean,full_matrices=False);pca_codes=u[:,:R]*s[:R];pca_basis=vh[:R];pca=((pca_codes@pca_basis+mean)>0).float()
   # Stronger generic control: fit rank-R logistic matrix factorization directly to task masks.
   t0=time.perf_counter();gc=(u[:,:R]*s[:R].clamp_min(1e-4).sqrt()).detach().requires_grad_();gb=(vh[:R].T*s[:R].clamp_min(1e-4).sqrt()).detach().requires_grad_();opt=torch.optim.Adam([gc,gb],lr=.04)
   for _ in range(250):
    opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(gc@gb.T,target);loss.backward();opt.step()
   generic_fit_s=time.perf_counter()-t0;logistic=((gc.detach()@gb.detach().T)>0).float()
   methods={'packed_binary_mask':(target,None,pack_masks(masks,'packed_binary_mask')),'continuous_mirror':(mirror,None,pack_tensors('continuous_mirror',[mb,mc])),'generic_pca_mask':(pca,None,pack_tensors('generic_pca_mask',[mean.half(),pca_basis.half(),pca_codes.half()])), 'generic_logistic_rank2':(logistic,None,pack_tensors('generic_logistic_rank2',[gb.detach().half(),gc.detach().half()]))}
   for name,(pred,_,payload) in methods.items():
    acc=float((pred==target).float().mean());q=(x*pred[:,None,:]);y=x*target[:,None,:];qerr=float((q-y).norm()/y.norm().clamp_min(1e-12));overlap=acc
    rows.append({'phase':phase,'world':world,'seed':seed,'method':name,'mask_accuracy':acc,'query_nrmse':qerr,'payload_bytes':len(payload),'payload_sha256':hashlib.sha256(payload).hexdigest(),'active_mask_bits':int(pred.sum()),'generic_fit_seconds':generic_fit_s if name=='generic_logistic_rank2' else 0.0})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
