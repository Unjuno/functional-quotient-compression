#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"artifacts";SEEDS=(34101,34102,34111,34112,34113);DIN=8;DOUT=4;NCLIENT=64;NTRAIN=48;NSUPPORT=32;NTEST=512

def pack(arrays,meta):
 b=io.BytesIO()
 with zipfile.ZipFile(b,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for name in sorted(arrays):
   q=io.BytesIO();np.save(q,np.asarray(arrays[name]),allow_pickle=False);i=zipfile.ZipInfo(name+".npy",(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o600<<16;z.writestr(i,q.getvalue())
  i=zipfile.ZipInfo("metadata.json",(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o600<<16;z.writestr(i,json.dumps(meta,sort_keys=True,separators=(",",":")).encode())
 data=b.getvalue()
 with zipfile.ZipFile(io.BytesIO(data)) as z:
  a={k[:-4]:np.load(io.BytesIO(z.read(k)),allow_pickle=False) for k in z.namelist() if k.endswith(".npy")};m=json.loads(z.read("metadata.json"))
 return data,a,m

def nerr(p,y):return float(np.mean((p-y)**2)/(np.mean(y*y)+1e-30))

def world(seed):
 rng=np.random.default_rng(seed);w0=rng.normal(size=(DOUT,DIN)).astype(np.float32)*.22;z=rng.normal(size=(DOUT*DIN,2));q,_=np.linalg.qr(z);U,V=q.T.reshape(2,DOUT,DIN).astype(np.float32);amp=.3;offset=rng.uniform(0,2*np.pi)
 theta=(offset+2*np.pi*np.arange(NCLIENT)/NCLIENT).astype(np.float32);features=np.stack([np.cos(theta),np.sin(theta)],1).astype(np.float32)
 targets=np.stack([w0+amp*(f[0]*U+f[1]*V) for f in features]);off=(w0+rng.normal(size=(DOUT,DIN)).astype(np.float32)*.4)
 support=[];truth=[];xeval=[]
 for i in range(NTRAIN):
  x=rng.normal(size=(NSUPPORT,DIN)).astype(np.float32);y=x@targets[i].T;support.append((x,y))
 for i in range(NCLIENT):
  x=rng.normal(size=(NTEST,DIN)).astype(np.float32);truth.append(x@targets[i].T);xeval.append(x)
 xoff=rng.normal(size=(NSUPPORT,DIN)).astype(np.float32);yoff=xoff@off.T;xofftest=rng.normal(size=(NTEST,DIN)).astype(np.float32);yofftest=xofftest@off.T
 return w0,U,V,amp,features,targets,off,support,truth,xeval,(xoff,yoff,xofftest,yofftest)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dev-only',action='store_true');args=ap.parse_args();OUT.mkdir(exist_ok=True);rows=[]
 for seed in SEEDS[:2] if args.dev_only else SEEDS:
  w0,U,V,amp,z,targets,off,support,truth,xeval,offdata=world(seed)
  # Estimate each observed client's full matrix from its local support examples.
  estimates=[]
  for x,y in support:estimates.append(np.linalg.lstsq(x,y,rcond=None)[0].T.astype(np.float32))
  design=np.column_stack([np.ones(NTRAIN),z[:NTRAIN]])
  gen=np.linalg.lstsq(design,np.stack(estimates).reshape(NTRAIN,-1),rcond=None)[0].astype(np.float32)
  generated=(np.column_stack([np.ones(NCLIENT),z])@gen).reshape(NCLIENT,DOUT,DIN)
  phase=np.arctan2(z[:,1],z[:,0]).astype(np.float32)
  zoff=np.array([[1.,0.]],np.float32);w_off_pred=(np.column_stack([np.ones(1),zoff])@gen).reshape(DOUT,DIN);residual=(off-w_off_pred).astype(np.float32)
  fedavg=np.mean(estimates,axis=0).astype(np.float32)
  cases={
   'independent_65':({f'client_{i}':m for i,m in enumerate(list(targets)+[off])},{'kind':'independent','clients':65}),
   'fedavg_shared':({'weight':fedavg},{'kind':'fedavg'}),
   'pfedhn_linear_full_model':({'generator':gen,'client_embeddings':np.vstack([z,zoff]).astype(np.float32)},{'kind':'linear_full_model_hypernetwork','clients':65,'private':False}),
   'pfedhn_with_private':({'generator':gen,'client_embeddings':np.vstack([z,zoff]).astype(np.float32),'private_residual':residual},{'kind':'linear_full_model_hypernetwork','clients':65,'private':True}),
   'mirror_phase':({'generator':gen,'client_phase':np.r_[phase,0.].astype(np.float32)},{'kind':'mirror_phase','clients':65,'private':False}),
   'mirror_phase_with_private':({'generator':gen,'client_phase':np.r_[phase,0.].astype(np.float32),'private_residual':residual},{'kind':'mirror_phase','clients':65,'private':True}),
   'native_scalar_phase_with_private':({'generator':gen,'client_phase':np.r_[phase,0.].astype(np.float32),'private_residual':residual},{'kind':'mirror_phase','clients':65,'private':True})}
  for name,(arrays,meta) in cases.items():
   t0=time.perf_counter();payload,ld,md=pack(arrays,meta)
   if name=='independent_65':predW=[ld[f'client_{i}'] for i in range(NCLIENT)]+[ld['client_64']]
   elif name=='fedavg_shared':predW=[ld['weight'] for _ in range(NCLIENT+1)]
   elif 'pfedhn' in name:
    coords=ld['client_embeddings'];predW=[(np.r_[1.,coords[i]]@ld['generator']).reshape(DOUT,DIN) for i in range(65)]
    if 'private_residual' in ld:predW[-1]=predW[-1]+ld['private_residual']
   else:
    ph=ld['client_phase'];predW=[ld['generator'][0].reshape(DOUT,DIN)+np.cos(ph[i])*ld['generator'][1].reshape(DOUT,DIN)+np.sin(ph[i])*ld['generator'][2].reshape(DOUT,DIN) for i in range(65)]
    if 'private_residual' in ld:predW[-1]=predW[-1]+ld['private_residual']
   errs=[nerr(xeval[i]@predW[i].T,truth[i]) for i in range(NCLIENT)];offpred=offdata[2]@predW[-1].T;offerr=nerr(offpred,offdata[3]);wall=time.perf_counter()-t0
   rows.append({'seed':seed,'method':name,'payload_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'seen_client_mean_nMSE':float(np.mean(errs[:NTRAIN])),'unseen_client_mean_nMSE':float(np.mean(errs[NTRAIN:])),'unseen_client_max_nMSE':max(errs[NTRAIN:]),'offorbit_nMSE':offerr,'communication_bytes_per_unseen_client':8 if 'pfedhn' in name else 4,'examples_seen':NTRAIN*NSUPPORT,'heldout_examples':NCLIENT*NTEST,'optimizer_updates':0,'fit_compute_proxy_MACs':int(NTRAIN*DIN*DIN+NTRAIN*DIN*DOUT+NTRAIN*3*DOUT*DIN),'generation_MACs_64_clients':int(NCLIENT*3*DOUT*DIN),'mirror_transform_MACs_64_clients':int(NCLIENT*2*DOUT*DIN),'wall_seconds_serialize_inference':wall})
 out=OUT/('development.csv' if args.dev_only else 'results.csv')
 with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
 for r in rows:print(json.dumps(r,sort_keys=True))
if __name__=='__main__':main()
