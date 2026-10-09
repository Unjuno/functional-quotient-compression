"""MA-296 synthetic task-vector superposition retrieval screen."""
import argparse,csv,hashlib,json,struct,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; D,N=512,8; DEV=[29600,29601]; FRESH=[29610,29611,29612]; SEEDS=[0,1,2]
torch.set_num_threads(2)
def make(world,seed):
 g=torch.Generator().manual_seed(world*100003+seed*7919+296)
 shared=torch.randn(2,D,generator=g); private=torch.randn(N,D,generator=g)
 shared=shared/torch.linalg.vector_norm(shared,dim=1,keepdim=True); private=private/torch.linalg.vector_norm(private,dim=1,keepdim=True)
 # Correlated task edits induce interference under unbound superposition.
 coeff=torch.randn(N,2,generator=g)*1.5
 deltas=coeff@shared+0.35*private
 keys=torch.randint(0,2,(N,N),generator=g,dtype=torch.int64).float()*2-1
 keys=keys/keys.norm(dim=1,keepdim=True)
 had=torch.tensor([[1. if (i&j).bit_count()%2==0 else -1. for j in range(N)] for i in range(N)])/N**.5
 x=torch.randn(N,768,D,generator=g); y=torch.einsum('ntd,nd->nt',x,deltas)
 return deltas,keys,had,x,y
def pack(parts,method):
 meta=json.dumps({'method':method,'shapes':[list(p.shape) for p in parts],'dtype':'float32','D':D,'N':N},sort_keys=True,separators=(',',':')).encode()
 raw=b''.join(p.detach().cpu().float().contiguous().numpy().tobytes() for p in parts)
 return b'MA296\0'+struct.pack('<I',len(meta))+meta+raw
def nrmse(a,b): return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-12))
def run(phase):
 rows=[]
 for world in DEV if phase=='development' else FRESH:
  for seed in SEEDS:
   ds,keys,had,x,y=make(world,seed); targets=[]
   # Decompose task edits into disjoint support/query examples; estimate task deltas by LS.
   estimates=[]
   for i in range(N):
    a=torch.linalg.lstsq(x[i,:640],y[i,:640]).solution; estimates.append(a)
   estimates=torch.stack(estimates)
   # Encode once, retrieve using task address. Addresses are paid in serialized payload.
   methods={
    'unbound_sum':(estimates.sum(0).expand_as(ds),[estimates.sum(0)]),
    'raw_independent':(estimates,[estimates]),
    'psp_random':(keys@torch.stack([(estimates*keys[:,j,None]).sum(0) for j in range(N)]),[torch.stack([(estimates*keys[:,j,None]).sum(0) for j in range(N)]),keys]),
    'mirror_hadamard':(had@torch.stack([(estimates*had[:,j,None]).sum(0) for j in range(N)]),[torch.stack([(estimates*had[:,j,None]).sum(0) for j in range(N)]),had]),
   }
   # Generic orthogonal task basis compression is a mandatory simple control.
   q,r=torch.linalg.qr(estimates.T,mode='reduced'); coeff=estimates@q
   methods['generic_orthogonal_basis']=(coeff@q.T,[q,coeff])
   for method,(pred,parts) in methods.items():
    start=time.perf_counter(); blob=pack(parts,method); elapsed=time.perf_counter()-start
    truth=ds; err=nrmse(pred,truth); xq=x[:,640:]; yq=torch.einsum('ntd,nd->nt',xq,pred)
    qerr=nrmse(yq,y[:,640:]); gram=pred@pred.T; interference=float(((gram-torch.diag(torch.diag(gram)))**2).sum().sqrt())
    rows.append({'phase':phase,'world':world,'seed':seed,'method':method,'nrmse':err,'query_nrmse':qerr,'interference':interference,'payload_bytes':len(blob),'fit_seconds':elapsed,'sha256':hashlib.sha256(blob).hexdigest()})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
