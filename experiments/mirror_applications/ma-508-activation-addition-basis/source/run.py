#!/usr/bin/env python3
"""MA-508 synthetic activation-addition bank and shared-basis screen."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from pathlib import Path
import numpy as np,torch
D,T,Q,TRUE_R,NTR,NTE=64,32,16,8,64,256
RANKS=(2,4,8,16);RHOS=(0.,.1,.25,.5);HELD=tuple(range(24,32))
METHODS=('explicit_vector','no_intervention','shared_mirror','native_pca')

def world(seed,rho):
 g=torch.Generator().manual_seed(seed+508);u,_=torch.linalg.qr(torch.randn(D,TRUE_R,generator=g));
 cg=torch.Generator().manual_seed(seed+1508);c=torch.randn(T,TRUE_R,generator=cg)*.2
 pg=torch.Generator().manual_seed(seed+2508);p=torch.randn(T,D,generator=pg)
 common=(u@c.T).T
 private=[]
 for t in range(T):
  z=p[t];z=z/z.norm().clamp_min(1e-12)*common[:,t].norm().clamp_min(1e-12)*rho;private.append(z)
 vec=common+torch.stack(private)
 qg=torch.Generator().manual_seed(seed+3508);probe,_=torch.linalg.qr(torch.randn(D,Q,generator=qg));probe=probe.T
 xg=torch.Generator().manual_seed(seed+4508);xtr=torch.randn(T,NTR,D,generator=xg);xte=torch.randn(T,NTE,D,generator=xg)
 # Activation-addition contrast vectors are directly observable from h+s minus h.
 ytr=xtr+vec[:,None,:];yte=xte+vec[:,None,:]
 support=(ytr-xtr).mean(1)
 return {'vectors':vec,'probe':probe,'xtr':xtr,'ytr':ytr,'xte':xte,'yte':yte,'support':support}

def basis_fit(w,rank):
 start=time.perf_counter();train=w['support'][:24];_,_,vh=torch.linalg.svd(train,full_matrices=False);b=vh[:rank].T
 codes=w['support']@b
 return b,codes,time.perf_counter()-start

def objects(method,w,rank):
 if method=='explicit_vector':return {'probe':w['probe'],'vectors':w['support']},0.
 if method=='no_intervention':return {'probe':w['probe']},0.
 b,c,wall=basis_fit(w,rank)
 return {'probe':w['probe'],'basis':b,'codes':c},wall

def pack(method,obj,rank):
 family='native_pca_activation_addition' if method in ('shared_mirror','native_pca') else method
 schema={'format':'MA508-activation-addition-v1','method_class':family,'hidden_dimension':D,'behaviors':T,'rank':rank if 'basis' in obj else None,'probe_outputs':Q}
 arr={k:v.detach().cpu().numpy().astype(np.float32) for k,v in obj.items()};arr['schema_json']=np.frombuffer(json.dumps(schema,sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8)
 bio=io.BytesIO();np.savez(bio,**arr);return bio.getvalue()

def add_vector(method,obj,t,x):
 if method=='explicit_vector':return x+obj['vectors'][t]
 if method=='no_intervention':return x
 if method in ('shared_mirror','native_pca'):return x+obj['basis']@obj['codes'][t]
 raise ValueError(method)

def score(method,obj,w):
 yhat=[];start=time.perf_counter()
 for t in range(T):yhat.append(w['xte'][t]+(add_vector(method,obj,t,w['xte'][t])-w['xte'][t]))
 wall=time.perf_counter()-start;yhat=torch.stack(yhat);target=w['yte'];probe=obj['probe'];zhat=torch.einsum('qd,tnd->tnq',probe,yhat);zref=torch.einsum('qd,tnd->tnq',probe,target);err=zhat-zref
 signal=(zref-torch.einsum('qd,tnd->tnq',probe,w['xte'])).square().mean().sqrt().clamp_min(1e-12)
 held=torch.tensor(HELD);rel=float(err[held].square().mean().sqrt()/((zref[held]-torch.einsum('qd,tnd->tnq',probe,w['xte'][held])).square().mean().sqrt().clamp_min(1e-12)))
 off=[]
 for t in HELD:
  mask=torch.arange(Q)!=t%Q;off.append(err[t,:,mask])
 offrmse=float(torch.cat([a.reshape(-1) for a in off]).square().mean().sqrt())
 if method in ('shared_mirror','native_pca'):
  vhat=(obj['basis']@obj['codes'].T).T;unique=int(torch.unique(torch.round(vhat[held]*1e6),dim=0).shape[0]);zero=w['xte'][held];causal=float((yhat[held]-zero).abs().max());ops=D*obj['basis'].shape[1]+D
 elif method=='explicit_vector':vhat=obj['vectors'];unique=int(torch.unique(torch.round(vhat[held]*1e6),dim=0).shape[0]);causal=float(vhat[held].abs().max());ops=D
 else:vhat=torch.zeros(T,D);unique=0;causal=0.;ops=0
 return {'heldout_probe_relative_rmse':rel,'heldout_off_target_probe_rmse':offrmse,'heldout_behavior_count':len(HELD),'unique_heldout_views':unique,'max_output_change_when_code_zeroed':causal,'active_compute_proxy_per_example':ops,'active_compute_proxy_all_eval_examples':ops*T*NTE,'inference_wall_s':wall}

def run(seed,out):
 out.mkdir(parents=True,exist_ok=True);doc={'experiment_id':'MA-508','seed':seed,'split':'dev','ranks':{}}
 for rho in RHOS:
  w=world(seed,rho);rd=out/f'rho_{rho:g}';rd.mkdir(exist_ok=True);byrank={}
  for rank in RANKS:
   methods={};cache={}
   for method in METHODS:
    key='shared_mirror' if method=='native_pca' else method
    if key not in cache:
     start=time.perf_counter();obj,fitwall=objects(key,w,rank);raw=pack(key,obj,rank);cache[key]=(obj,raw,fitwall)
    obj,raw,fitwall=cache[key];(rd/f'rank_{rank}_{method}_payload.npz').write_bytes(raw)
    met=score(key,obj,w);met.update({'inference_payload_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'fit_wall_s':fitwall,'optimizer_updates':0,'calibration_contrasts':T*NTR,'basis_fit_behavior_count':24})
    methods[method]=met
   assert (rd/f'rank_{rank}_shared_mirror_payload.npz').read_bytes()==(rd/f'rank_{rank}_native_pca_payload.npz').read_bytes()
   byrank[str(rank)]=methods
  doc['ranks'][str(rho)]=byrank
 (out/'metrics.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');return doc

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.seed,a.out),sort_keys=True))
if __name__=='__main__':main()
