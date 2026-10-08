#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"artifacts";SEEDS=(34201,34202,34211,34212,34213);D=16;R=2;NCLIENT=64;NTRAIN=48;NS=40;NT=512

def pack(arrays,meta):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for n,a in sorted(arrays.items()):
   q=io.BytesIO();np.save(q,np.asarray(a),allow_pickle=False);i=zipfile.ZipInfo(n+'.npy',(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o600<<16;z.writestr(i,q.getvalue())
  i=zipfile.ZipInfo('metadata.json',(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o600<<16;z.writestr(i,json.dumps(meta,sort_keys=True,separators=(',',':')).encode())
 payload=b.getvalue()
 with zipfile.ZipFile(io.BytesIO(payload)) as z:
  aa={n[:-4]:np.load(io.BytesIO(z.read(n)),allow_pickle=False) for n in z.namelist() if n.endswith('.npy')};mm=json.loads(z.read('metadata.json'))
 return payload,aa,mm

def nerr(a,b):return float(np.mean((a-b)**2)/(np.mean(b*b)+1e-30))

def world(seed):
 rng=np.random.default_rng(seed);W0=rng.normal(size=(D,D)).astype(np.float32)*.1
 b=rng.normal(size=(D,R)).astype(np.float32);b,_=np.linalg.qr(b);a1=rng.normal(size=(R,D)).astype(np.float32)*.12;a2=rng.normal(size=(R,D)).astype(np.float32)*.12;offset=rng.uniform(0,2*np.pi)
 theta=(offset+2*np.pi*np.arange(NCLIENT)/NCLIENT).astype(np.float32);z=np.stack([np.cos(theta),np.sin(theta)],1).astype(np.float32)
 target=np.stack([W0+b@(np.cos(t)*a1+np.sin(t)*a2) for t in theta])
 # Off-orbit client shares B but has a right factor outside the two-dimensional client orbit.
 aoff=rng.normal(size=(R,D)).astype(np.float32)*.25;zoff=np.array([[1.,0.]],np.float32);off=W0+b@aoff
 support=[];xev=[];truth=[]
 for i in range(NTRAIN):
  x=rng.normal(size=(NS,D)).astype(np.float32);support.append((x,x@target[i].T))
 for i in range(NCLIENT):
  x=rng.normal(size=(NT,D)).astype(np.float32);xev.append(x);truth.append(x@target[i].T)
 xo=rng.normal(size=(NS,D)).astype(np.float32);yo=xo@off.T;xt=rng.normal(size=(NT,D)).astype(np.float32);yt=xt@off.T
 return W0,b,a1,a2,aoff,z,zoff,target,off,support,xev,truth,(xo,yo,xt,yt)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dev-only',action='store_true');args=ap.parse_args();OUT.mkdir(exist_ok=True);rows=[]
 for seed in SEEDS[:2] if args.dev_only else SEEDS:
  W0,b0,a10,a20,aoff0,z,zoff,target,off,support,xev,truth,offdata=world(seed)
  # Estimate seen full matrices from client examples, then fit the shared low-rank adapter family.
  estimates=np.stack([np.linalg.lstsq(x,y,rcond=None)[0].T.astype(np.float32) for x,y in support])
  design=np.column_stack([np.ones(NTRAIN),z[:NTRAIN]])
  coef=np.linalg.lstsq(design,estimates.reshape(NTRAIN,-1),rcond=None)[0].astype(np.float32)
  base=coef[0].reshape(D,D);d1=coef[1].reshape(D,D);d2=coef[2].reshape(D,D)
  # Recover one shared left factor from the span of the two learned functional directions.
  u,s,_=np.linalg.svd(np.concatenate([d1,d2],axis=1),full_matrices=False);B=u[:,:R].astype(np.float32);A1=(B.T@d1).astype(np.float32);A2=(B.T@d2).astype(np.float32)
  # Local off-orbit task estimate is represented as a private right-factor correction over shared B.
  aoff=np.linalg.lstsq(B,off-base,rcond=None)[0].astype(np.float32);pred_off=base+B@A1;# Mirror's default z_off=(1,0)
  private_A=(aoff-A1).astype(np.float32)
  # HyperLoRA generator maps [1,z1,z2] to complete B,A factors. B is emitted too, not treated as free.
  gen=np.zeros((3,R*D+R*D),np.float32);gen[0,:R*D]=B.reshape(-1);gen[1,R*D:]=A1.reshape(-1);gen[2,R*D:]=A2.reshape(-1)
  phase=np.arctan2(z[:,1],z[:,0]).astype(np.float32);phase_all=np.r_[phase,0.].astype(np.float32);zall=np.vstack([z,zoff]).astype(np.float32)
  priv_arrays={'private_A':private_A}
  cases={
   'independent_65_lora':({**{f'B{i}':np.linalg.svd(target[i]-base,full_matrices=False)[0][:,:R].astype(np.float32) for i in range(NCLIENT)},**{f'A{i}':(np.linalg.svd(target[i]-base,full_matrices=False)[1][:R,None]*np.linalg.svd(target[i]-base,full_matrices=False)[2][:R]).astype(np.float32) for i in range(NCLIENT)},'B64':B,'A64':aoff,'base':base},{'kind':'independent_rank2_lora','count':65}),
   'fedavg_adapter':({'weight':np.mean(estimates,axis=0).astype(np.float32)},{'kind':'fedavg'}),
   'hyperlora_full_factor_generator':({'base':base,'generator':gen,'client_embeddings':zall},{'kind':'hyperlora_linear_full_AB','private':False}),
   'hyperlora_with_private':({'base':base,'generator':gen,'client_embeddings':zall,'private_A':private_A},{'kind':'hyperlora_linear_full_AB','private':True}),
   'mirror_phase':({'base':base,'B':B,'A1':A1,'A2':A2,'client_phase':phase_all},{'kind':'mirror_phase','private':False}),
   'mirror_phase_with_private':({'base':base,'B':B,'A1':A1,'A2':A2,'client_phase':phase_all,'private_A':private_A},{'kind':'mirror_phase','private':True}),
   'direct_coefficients_with_private':({'base':base,'B':B,'A1':A1,'A2':A2,'client_coefficients':np.vstack([z,zoff]),'private_A':private_A},{'kind':'direct_coefficients','private':True}),
   'native_phase_with_private':({'base':base,'B':B,'A1':A1,'A2':A2,'client_phase':phase_all,'private_A':private_A},{'kind':'mirror_phase','private':True})}
  for name,(arrays,meta) in cases.items():
   t0=time.perf_counter();payload,ld,md=pack(arrays,meta)
   if name=='independent_65_lora':
    predW=[]
    for i in range(NCLIENT):predW.append(ld['base']+ld[f'B{i}']@ld[f'A{i}'])
    predW.append(ld['base']+ld['B64']@ld['A64'])
   elif name=='fedavg_adapter':predW=[ld['weight'] for _ in range(NCLIENT+1)]
   elif name.startswith('hyperlora'):
    predW=[]
    for i in range(NCLIENT+1):
     f=np.r_[1.,ld['client_embeddings'][i]]@ld['generator'];Bi=f[:R*D].reshape(D,R);Ai=f[R*D:].reshape(R,D);W=ld['base']+Bi@Ai
     if i==NCLIENT and 'private_A' in ld:W+=ld['generator'][0,:R*D].reshape(D,R)@ld['private_A']
     predW.append(W)
   else:
    predW=[]
    for i in range(NCLIENT+1):
     if 'client_coefficients' in ld:c1,c2=ld['client_coefficients'][i]
     else:c1,c2=np.cos(ld['client_phase'][i]),np.sin(ld['client_phase'][i])
     W=ld['base']+ld['B']@(c1*ld['A1']+c2*ld['A2'])
     if i==NCLIENT and 'private_A' in ld:W+=ld['B']@ld['private_A']
     predW.append(W)
   ee=[nerr(xev[i]@predW[i].T,truth[i]) for i in range(NCLIENT)];offerr=nerr(offdata[2]@predW[-1].T,offdata[3]);wall=time.perf_counter()-t0
   rows.append({'seed':seed,'method':name,'payload_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'unseen_client_mean_nMSE':float(np.mean(ee[NTRAIN:])),'unseen_client_max_nMSE':max(ee[NTRAIN:]),'offorbit_nMSE':offerr,'client_code_bytes':8 if name.startswith('hyperlora') or 'coefficients' in name else 4,'support_examples':NTRAIN*NS+NS,'heldout_examples':NCLIENT*NT,'optimizer_updates':0,'basis_fit_MAC_proxy':int(NTRAIN*D*D*D),'generation_MACs_64_clients':int(NCLIENT*3*(R*D+R*D)) if name.startswith('hyperlora') else int(NCLIENT*R*D*2),'inference_adapter_MACs_per_token':int(R*D*2),'wall_seconds_serialization_eval':wall})
 out=OUT/('development.csv' if args.dev_only else 'results.csv')
 with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
 for r in rows:print(json.dumps(r,sort_keys=True))
if __name__=='__main__':main()
