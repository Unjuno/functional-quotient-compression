"""MA-292: task-vector compression and held-out additive compositions."""
import argparse,csv,hashlib,json,math,struct,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];D,N,K=256,16,4;DEV,FRESH=[29200,29201],[29210,29211,29212];SEEDS=[0,1,2]
torch.set_num_threads(2)
def make(world,seed):
 g=torch.Generator().manual_seed(world*100003+seed*7919+292);basis=torch.randn(K,D,generator=g);basis=torch.linalg.qr(basis.T).Q.T
 coeff=torch.randn(N,K,generator=g);deltas=coeff@basis
 xtrain=torch.randn(N,64,D,generator=g);xquery=torch.randn(N,128,D,generator=g);ytrain=torch.einsum('ntd,nd->nt',xtrain,deltas);yquery=torch.einsum('ntd,nd->nt',xquery,deltas)
 train=list(range(12));held=list(range(12,16));pairs=[(12,13),(12,14),(13,15),(14,15),(0,12),(1,13),(2,14),(3,15)]
 return basis,coeff,deltas,xtrain,ytrain,xquery,yquery,train,held,pairs
def pack(parts,method):
 flat=torch.cat([p.flatten() for p in parts]).float().numpy().tobytes();meta=json.dumps({'method':method,'shapes':[list(p.shape) for p in parts],'dim':D},sort_keys=True,separators=(',',':')).encode();return b'MA292\0'+struct.pack('<I',len(meta))+meta+flat
def nrmse(a,b):return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-12))
def run(phase):
 rows=[]
 for world in (DEV if phase=='development' else FRESH):
  for seed in SEEDS:
   basis,coeff,deltas,xtrain,ytrain,xquery,yquery,train,held,pairs=make(world,seed)
   # fit PCA on train task deltas only
   _,s,vh=torch.linalg.svd(deltas[train],full_matrices=False);pca=vh[:K]
   train_codes=deltas[train]@pca.T
   for split,ids in [('heldout_tasks',held),('heldout_compositions',range(len(pairs)))]:
    targets=[];queries=[]
    if split=='heldout_tasks':
     targets=[deltas[i] for i in ids];queries=[(xtrain[i],ytrain[i],xquery[i],yquery[i]) for i in ids]
    else:
     targets=[deltas[i]+deltas[j] for i,j in pairs];queries=[(xtrain[i],xtrain[i]@targets[n],xquery[i],xquery[i]@targets[n]) for n,(i,j) in enumerate(pairs)]
    for method in ['raw_vectors','pca_basis','mirror_basis','generic_coeff','task_addition','independent_upper']:
     start=time.perf_counter()
     if method=='raw_vectors':pred=torch.stack(targets);parts=[torch.stack(targets)]
     elif method=='independent_upper':pred=torch.stack(targets);parts=[torch.stack(targets)]
     elif method in ('pca_basis','mirror_basis','generic_coeff'):
      mat=pca if method=='pca_basis' else basis[:K];train_code=deltas[train]@mat.T;preds=[]
      for n,y in enumerate(targets):
       xs,ys,xq,yq=queries[n]
       # Fit basis coefficients only from support input/output observations.
       design=torch.einsum('nd,kd->nk',xs,mat);code=torch.linalg.lstsq(design,ys).solution;preds.append(code@mat)
      pred=torch.stack(preds);parts=[mat,train_code]
     elif method=='task_addition':
      pred=torch.stack([deltas[i]+deltas[j] for i,j in pairs]) if split=='heldout_compositions' else torch.stack([deltas[i] for i in ids]);parts=[deltas[train]]
     sec=time.perf_counter()-start;truth=torch.stack(targets);err=nrmse(pred,truth);inter=float((pred-truth).square().mean().sqrt());blob=pack(parts,method);path=ROOT/'artifacts'/'payloads'/f'{world}_{seed}_{split}_{method}.bin';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob)
     rows.append({'phase':phase,'world':world,'seed':seed,'split':split,'method':method,'nrmse':err,'payload_bytes':len(blob),'interference':inter,'fit_seconds':sec,'sha256':hashlib.sha256(blob).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
