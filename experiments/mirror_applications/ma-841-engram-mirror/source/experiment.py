"""MA-841 closed-form synthetic Engram trace compression screen."""
from __future__ import annotations
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(1);D=8;R=4

def make_world(seed):
 rng=np.random.default_rng(seed);q,_=np.linalg.qr(rng.normal(size=(D*D,R)));true=q[:,:R].T
 def trace(aligned,idx):
  coeff=rng.normal(0,1,(R,));flat=coeff@true
  if not aligned:flat=flat+.45*rng.normal(size=D*D)
  m=flat.reshape(D,D);K=rng.normal(size=(D,D));K+=np.eye(D)*.5;V=K@m
  # Closed-form ridge/native Engram trace from key-value constraints.
  est=np.linalg.solve(K.T@K+1e-5*np.eye(D),K.T@V).astype('float32')
  return {'aligned':aligned,'M':est,'K':K.astype('float32'),'V':V.astype('float32'),'id':idx}
 dev=[trace(True,f'dev{i}') for i in range(8)]
 mat=np.stack([t['M'].reshape(-1) for t in dev]);_,_,vt=np.linalg.svd(mat,full_matrices=False);basis=vt[:R].astype('float32')
 audit=[trace(True,f'low{i}') for i in range(16)]+[trace(False,f'private{i}') for i in range(8)]
 return dev,basis,audit

def project(M,B):
 v=M.reshape(-1);c=v@B.T;rec=(c@B).reshape(D,D);return c.astype('float32'),rec.astype('float32')
def causal(M,ref,K,V,otherK):
 pred=K@M;refpred=K@ref;den=np.mean(V*V)+1e-9
 suff=float(np.mean((pred-V)**2)/den)
 cos=float((pred.ravel()@V.ravel())/(np.linalg.norm(pred)*np.linalg.norm(V)+1e-9))
 leak=float(np.mean((otherK@M)**2)/(np.mean((otherK@ref)**2)+1e-9))
 necessity=float(np.linalg.norm(pred)/(np.linalg.norm(refpred)+1e-9))
 return suff,cos,leak,necessity

def serialize(method,B,codes,traces,residuals,indices,path):
 if method=='native':data={'trace_bank':torch.tensor(traces).contiguous()}
 elif method=='random':data={'basis':torch.tensor(B).contiguous(),'code_bank':torch.tensor(codes).contiguous()}
 elif method=='private':
  data={'basis':torch.tensor(B).contiguous(),'code_bank':torch.tensor(codes).contiguous()}
  if len(indices):data['private_residuals']=torch.tensor(residuals).contiguous();data['private_indices']=torch.tensor(indices,dtype=torch.int64)
 else:data={'basis':torch.tensor(B).contiguous(),'code_bank':torch.tensor(codes).contiguous()}
 save_file(data,str(path));raw=path.read_bytes();return len(raw),hashlib.sha256(raw).hexdigest()

def main():
 cfg=json.loads((ROOT/'PROTOCOL.json').read_text());allrows=[]
 for seed in cfg['fresh']['worlds_or_seeds']:
  dev,B,audit=make_world(seed);rng=np.random.default_rng(seed+10);rand,_=np.linalg.qr(rng.normal(size=(D*D,R)));rand=rand.T.astype('float32')
  native=[];mir=[];pca=[];random=[];hybrid=[];codes=[];resids=[];indices=[];rows=[];start=time.perf_counter()
  for i,t in enumerate(audit):
   c,mhat=project(t['M'],B);_,phat=project(t['M'],B);_,rhat=project(t['M'],rand)
   residual=t['M']-mhat;err=float(np.mean(residual**2));
   if err>1e-4:indices.append(i);resids.append(residual)
   hhat=mhat+(residual if err>1e-4 else 0)
   otherK=np.concatenate([x['K'] for j,x in enumerate(audit) if j!=i][:3],axis=0)
   for method,M in [('native',t['M']),('mirror',mhat),('pca',phat),('random',rhat),('private',hhat)]:
    su,co,sp,ne=causal(M,t['M'],t['K'],t['V'],otherK)
    rows.append({'world':seed,'memory':i,'stratum':'aligned' if t['aligned'] else 'residual','method':method,'sufficiency_norm_mse':su,'reactivation_cosine':co,'specificity_leakage_ratio':sp,'necessity_effect_ratio':ne,'trace_reconstruction_mse':float(np.mean((M-t['M'])**2)),'examples':8,'updates':0,'active_macs_per_query':2*D*D,'train_seconds':0.0})
   native.append(t['M']);codes.append(c);mir.append(mhat);pca.append(phat);random.append(rhat);hybrid.append(hhat)
  all_native=np.stack(native);all_codes=np.stack(codes);all_res=np.stack(resids) if resids else np.zeros((0,D,D),dtype='float32')
  payloads={}
  for method,Buse,C,T,Res,Ids in [('native',B,all_codes,all_native,all_res,indices),('mirror',B,all_codes,all_native,all_res,indices),('pca',B,all_codes,all_native,all_res,indices),('random',rand,all_codes,all_native,all_res,indices),('private',B,all_codes,all_native,all_res,indices)]:
   name={'mirror':'mirror','pca':'pca','random':'random','native':'native','private':'private'}[method];path=ROOT/'source'/f'.{name}-{seed}.safetensors'
   enc={'mirror':'mirror','pca':'pca','random':'random','native':'native','private':'private'}[method]
   n,h=serialize(enc,Buse,C,T,Res,Ids,path);path.unlink();payloads[method]=(n,h)
   for row in rows:
    if row['method']==method:row['serialized_bytes']=n;row['payload_sha256']=h;row['bytes_per_memory']=n/len(audit)
  elapsed=time.perf_counter()-start
  for row in rows:row['world_seconds']=elapsed
  allrows.extend(rows)
  for st in ['aligned','residual']:
   print(seed,st,flush=True)
   for m in ['native','mirror','pca','random','private']:
    q=[r for r in rows if r['stratum']==st and r['method']==m]
    if q:print(m,'suff',np.mean([r['sufficiency_norm_mse'] for r in q]),'cos',np.mean([r['reactivation_cosine'] for r in q]),'spec',np.mean([r['specificity_leakage_ratio'] for r in q]),'nec',np.mean([r['necessity_effect_ratio'] for r in q]),'bytes',payloads[m][0],flush=True)
  print('private residuals',len(indices),'of',len(audit),flush=True)
 (ROOT/'source'/'audit_results.json').write_text(json.dumps(allrows,indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
if __name__=='__main__':main()
