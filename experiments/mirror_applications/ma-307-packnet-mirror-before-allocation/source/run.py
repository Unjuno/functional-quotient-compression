"""MA-307 task View before PackNet physical allocation (synthetic screen)."""
import argparse,csv,hashlib,json,struct
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];D,N,K,R=2048,12,128,3;DEV=[30700,30701];FRESH=[30710,30711,30712];SEEDS=[0,1,2];THRESH=.10
torch.set_num_threads(2)
def make(w,s):
 g=torch.Generator().manual_seed(w*100003+s*7919+307);idx=torch.randperm(D,generator=g)[:K]
 pos=torch.arange(K).float();base=torch.randn(K,generator=g);bc=base*torch.cos(2*torch.pi*pos/K);bs=base*torch.sin(2*torch.pi*pos/K)
 amp=torch.rand(N,generator=g)*.35+.15;phase=torch.rand(N,generator=g)*2*torch.pi
 codes=torch.stack([torch.ones(N),amp*torch.cos(phase),-amp*torch.sin(phase)],1)
 local=codes@torch.stack([base,bc,bs]);targets=torch.zeros(N,D);targets[:,idx]=local
 # Tasks 8-11 require new private corrections on 16 coordinates each.
 private_idx=[]
 for t in range(8,N):
  pi=idx[torch.randperm(K,generator=g)[:16]];pv=torch.randn(16,generator=g)*.5;targets[t,pi]+=pv;private_idx.append(pi)
 x=torch.randn(N,128,D,generator=g);y=torch.einsum('ntd,nd->nt',x,targets)
 return targets,idx,torch.stack([base,bc,bs]),codes,x,y

def serialize(name,parts,meta_extra=None):
 meta={'method':name,'shapes':[list(t.shape) for t in parts],'dtypes':[str(t.dtype) for t in parts]};meta.update(meta_extra or {});m=json.dumps(meta,sort_keys=True,separators=(',',':')).encode();return b'MA307\0'+struct.pack('<I',len(m))+m+b''.join(t.contiguous().numpy().tobytes() for t in parts)
def sparse(z,cut):
 z=torch.where(z.abs()>cut,z,torch.zeros_like(z));ii=torch.nonzero(z,as_tuple=False).to(torch.int16);vv=z[z!=0].half();return ii,vv
def nrmse(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-12))
def run(phase):
 rows=[]
 for w in DEV if phase=='development' else FRESH:
  for seed in SEEDS:
   targets,idx,teacher,codes,x,y=make(w,seed)
   # Mirror representation stores shared basis once, low-dim task codes, and only novel sparse residuals.
   b=teacher.half().float();mcode=codes.half().float();rec=torch.zeros_like(targets);rec[:,idx]=mcode@b;res=targets-rec;msplit=(res.norm(dim=1)/targets.norm(dim=1).clamp_min(1e-12))>THRESH;res[~msplit]=0;res=torch.where(res.abs()<1e-6,torch.zeros_like(res),res);ii,vv=sparse(res,.02);pred=rec.clone();
   if ii.numel():pred[ii[:,0].long(),ii[:,1].long()]+=vv.float()
   # Generic control: rank-3 PCA learned from first eight PackNet tasks, same sparse fallback policy.
   mu=targets[:8,idx].mean(0);u,sv,vh=torch.linalg.svd(targets[:8,idx]-mu,full_matrices=False);gb=vh[:R]
   # Generic robust shared-subspace fit: infer code from majority coordinates, then retain sparse outliers.
   gc=[];gresloc=[];gsplit=[]
   for t in range(N):
    loc=targets[t,idx]-mu;c=loc@gb.T;rr=loc-c@gb
    split_now=float(rr.norm()/loc.norm().clamp_min(1e-12))>THRESH;gsplit.append(split_now)
    if split_now:
     outlier=torch.topk(rr.abs(),16).indices;keep=torch.ones(K,dtype=torch.bool);keep[outlier]=False
     c=torch.linalg.lstsq(gb[:,keep].T,loc[keep]).solution;rr=loc-c@gb
    gc.append(c);gresloc.append(rr)
   gc=torch.stack(gc);gresloc=torch.stack(gresloc);gsplit_count=sum(gsplit);gbase=torch.zeros_like(targets);gbase[:,idx]=mu+gc@gb;gfullres=torch.zeros_like(targets);gfullres[:,idx]=gresloc;gii,gvv=sparse(gfullres,.02);gpred=gbase.clone()
   if gii.numel():gpred[gii[:,0].long(),gii[:,1].long()]+=gvv.float()
   # Keep ordinary PCA as a deliberately non-robust low-rank control too.
   mu0=targets[:8,idx].mean(0);pu,ps,pvh=torch.linalg.svd(targets[:8,idx]-mu0,full_matrices=False);pb=pvh[:R];pc=(targets[:,idx]-mu0)@pb.T;plocal=pc@pb+mu0;pbase=torch.zeros_like(targets);pbase[:,idx]=plocal;pres=targets-pbase;pii,pvv=sparse(pres,.02);ppred=pbase.clone()
   if pii.numel():ppred[pii[:,0].long(),pii[:,1].long()]+=pvv.float()
   # PackNet allocates a disjoint sparse set of physical values for every task.
   nz=torch.nonzero(targets!=0,as_tuple=False).to(torch.int16);vals=targets[targets!=0].half()
   methods={'packnet_private':(targets,serialize('packnet_private',[nz,vals],{'backbone_dim':D,'task_masks':N})),'mirror_before_split':(pred,serialize('mirror_before_split',[idx.to(torch.int16),b.half(),mcode.half(),ii,vv],{'threshold':THRESH})),'generic_robust_shared_basis':(gpred,serialize('generic_robust_shared_basis',[idx.to(torch.int16),mu.half(),gb.half(),gc.half(),gii,gvv],{'rank':R,'outlier_budget':16})), 'generic_pca_before_split':(ppred,serialize('generic_pca_before_split',[idx.to(torch.int16),mu0.half(),pb.half(),pc.half(),pii,pvv],{'rank':R})),'view_no_physical_split':(rec,serialize('view_no_physical_split',[idx.to(torch.int16),b.half(),mcode.half()],{'threshold':THRESH}))}
   for name,(out,payload) in methods.items():
    yp=torch.einsum('ntd,nd->nt',x,out);qerr=nrmse(yp,y);active=int((out.abs()>1e-6).sum());allocated=0 if name=='view_no_physical_split' else (int(ii.shape[0]) if name=='mirror_before_split' else (int((gii if name=='generic_robust_shared_basis' else pii).shape[0]) if name in ('generic_robust_shared_basis','generic_pca_before_split') else int(nz.shape[0])))
    rows.append({'phase':phase,'world':w,'seed':seed,'method':name,'query_nrmse':qerr,'payload_bytes':len(payload),'allocated_private_values':allocated,'private_split_tasks':(int(msplit.sum()) if name=='mirror_before_split' else (gsplit_count if name=='generic_robust_shared_basis' else (0 if name=='view_no_physical_split' else N))),'active_parameter_values':active,'payload_sha256':hashlib.sha256(payload).hexdigest()})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:wri=csv.DictWriter(f,fieldnames=rows[0]);wri.writeheader();wri.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
