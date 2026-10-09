import csv,hashlib,io,json,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];D=256;P=8;Q=8;SEEDS=[0,1,2];DEV=[25700,25701];FRESH=[25710,25711,25712];torch.set_num_threads(2)
def world(w,s):
 g=torch.Generator().manual_seed(w*100003+s*7919+257);A=torch.randn(P,D,generator=g);B=torch.randn(Q,D,generator=g);V=A[:,None,:]+B[None,:,:];mask=torch.tensor([[(i+j)%4!=0 for j in range(Q)] for i in range(P)]);return A,B,V,mask
def fit_factors(V,mask):
 pairs=mask.nonzero().tolist();X=torch.zeros(len(pairs),P+Q)
 for r,(i,j) in enumerate(pairs):X[r,i]=1;X[r,P+j]=1
 Y=torch.stack([V[i,j] for i,j in pairs]);sol=torch.linalg.lstsq(X,Y).solution;return sol[:P],sol[P:]
def radem(shape,seed):
 g=torch.Generator().manual_seed(seed);return torch.randint(0,2,shape,generator=g).float()*2-1
def payload(obj):b=io.BytesIO();torch.save(obj,b);return b.getvalue()
def nrmse(a,b):return float((a-b).square().mean().sqrt()/b.square().mean().sqrt())
def run(phase):
 rows=[];worlds=DEV if phase=='development' else FRESH
 for w in worlds:
  for s in SEEDS:
   A,B,V,mask=world(w,s);t=time.perf_counter();Ah,Bh=fit_factors(V,mask);fitsec=time.perf_counter()-t;trainpairs=mask.nonzero().tolist()
   Ct=radem((P*Q,D),w*100+s+11);S=torch.zeros(D)
   for i,j in trainpairs:S+=Ct[i*Q+j]*V[i,j]
   psp=torch.stack([S*Ct[i*Q+j] for i in range(P) for j in range(Q)]).reshape(P,Q,D)
   Ca=radem((P,D),w*100+s+21);Cb=radem((Q,D),w*100+s+31);Sa=(Ca*Ah).sum(0);Sb=(Cb*Bh).sum(0);mirA=Ca*Sa;mirB=Cb*Sb;mir=mirA[:,None,:]+mirB[None,:,:]
   fact=Ah[:,None,:]+Bh[None,:,:];oracle=A[:,None,:]+B[None,:,:]
   methods={'explicit_full':(V,{'tasks':V,'metadata':{'train_mask':mask,'shape':[P,Q,D]}}),'native_psp':(psp,{'tensor':S,'contexts':Ct,'metadata':{'train_pairs':trainpairs,'seed':w*100+s+11}}),'mirror_factorized':(mir,{'procedure_superposition':Sa,'domain_superposition':Sb,'procedure_contexts':Ca,'domain_contexts':Cb,'metadata':{'context_seeds':[w*100+s+21,w*100+s+31],'composition':'unbound then add'}}),'generic_factor_table':(fact,{'procedure_factors':Ah,'domain_factors':Bh,'metadata':{'decoder':'A_i+B_j'}}),'oracle_factor_table':(oracle,{'procedure_factors':A,'domain_factors':B,'metadata':{'oracle':True}})}
   for name,(pred,obj) in methods.items():
    blob=payload(obj);path=ROOT/'artifacts'/'payloads'/f'{w}_{s}_{name}.pt';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob)
    rows.append({'world':w,'seed':s,'method':name,'train_nrmse':nrmse(pred[mask],V[mask]),'heldout_nrmse':nrmse(pred[~mask],V[~mask]),'payload_bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'path':str(path.relative_to(ROOT.parents[2])),'factor_fit_seconds':fitsec,'task_count':P*Q,'train_combinations':int(mask.sum()),'heldout_combinations':int((~mask).sum())})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
