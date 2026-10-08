#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"artifacts";SEEDS=(34601,34602,34611,34612,34613);D=16;NC=64;NT=512;RATES=(0,.25,.5,.75,1.)

def pack(arrays,meta):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for n,a in sorted(arrays.items()):
   q=io.BytesIO();np.save(q,np.asarray(a),allow_pickle=False);i=zipfile.ZipInfo(n+'.npy',(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o600<<16;z.writestr(i,q.getvalue())
  i=zipfile.ZipInfo('metadata.json',(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o600<<16;z.writestr(i,json.dumps(meta,sort_keys=True,separators=(',',':')).encode())
 data=b.getvalue()
 with zipfile.ZipFile(io.BytesIO(data)) as z:
  aa={n[:-4]:np.load(io.BytesIO(z.read(n)),allow_pickle=False) for n in z.namelist() if n.endswith('.npy')};mm=json.loads(z.read('metadata.json'))
 return data,aa,mm

def nerr(a,b):return float(np.mean((a-b)**2)/(np.mean(b*b)+1e-30))

def world(seed):
 rng=np.random.default_rng(seed);W=rng.normal(size=(D,D)).astype(np.float32)*.1;U=rng.normal(size=(D,D)).astype(np.float32)*.035;V=rng.normal(size=(D,D)).astype(np.float32)*.035;ph=rng.uniform(-np.pi,np.pi,NC).astype(np.float32)
 common=np.stack([W+np.cos(t)*U+np.sin(t)*V for t in ph]);p=rng.normal(size=(NC,D)).astype(np.float32)*.08;q=rng.normal(size=(NC,D)).astype(np.float32)*.08;res=p[:,:,None]*q[:,None,:];
 x=[rng.normal(size=(NT,D)).astype(np.float32) for _ in range(NC)]
 return rng,W,U,V,ph,common,res,p,q,x

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dev-only',action='store_true');args=ap.parse_args();OUT.mkdir(exist_ok=True);rows=[]
 for seed in SEEDS[:2] if args.dev_only else SEEDS:
  rng,W,U,V,ph,common,res,p,q,x=world(seed)
  for rate in RATES:
   k=int(NC*rate);ids=np.arange(k,dtype=np.uint16);targets=common.copy();targets[:k]+=res[:k];
   cases={
    'independent_full':({'models':targets},{'kind':'independent','heterogeneity':rate}),
    'global_shared':({'weight':W},{'kind':'global_shared','heterogeneity':rate}),
    'mirror_no_private':({'base':W,'U':U,'V':V,'phase':ph},{'kind':'phase_shared','heterogeneity':rate,'private':False}),
    'mirror_rank1_private':({'base':W,'U':U,'V':V,'phase':ph,'private_ids':ids,'private_left':p[:k],'private_right':q[:k]},{'kind':'phase_shared_private_rank1','heterogeneity':rate,'private':True}),
    'direct_coeff_rank1_private':({'base':W,'U':U,'V':V,'coefficients':np.stack([np.cos(ph),np.sin(ph)],1),'private_ids':ids,'private_left':p[:k],'private_right':q[:k]},{'kind':'direct_coeff_private_rank1','heterogeneity':rate,'private':True}),
    'native_scalar_phase_private':({'base':W,'U':U,'V':V,'phase':ph,'private_ids':ids,'private_left':p[:k],'private_right':q[:k]},{'kind':'phase_shared_private_rank1','heterogeneity':rate,'private':True}),
    'fedrep_dense_private':({'base':W,'U':U,'V':V,'coefficients':np.stack([np.cos(ph),np.sin(ph)],1),'dense_private_ids':ids,'dense_residuals':res[:k]},{'kind':'fedrep_dense_private','heterogeneity':rate,'private':True})}
   for name,(arrays,meta) in cases.items():
    t0=time.perf_counter();payload,ld,md=pack(arrays,meta);pred=[]
    for i in range(NC):
     if name=='independent_full':m=ld['models'][i]
     elif name=='global_shared':m=ld['weight']
     elif name=='fedrep_dense_private':
      m=ld['base']+ld['coefficients'][i,0]*ld['U']+ld['coefficients'][i,1]*ld['V'];where=np.where(ld['dense_private_ids']==i)[0]
      if len(where):m=m+ld['dense_residuals'][where[0]]
     else:
      a,b=(np.cos(ld['phase'][i]),np.sin(ld['phase'][i])) if 'phase' in ld else ld['coefficients'][i]
      m=ld['base']+a*ld['U']+b*ld['V'];where=np.where(ld.get('private_ids',np.array([],np.uint16))==i)[0]
      if len(where):m=m+ld['private_left'][where[0],:,None]*ld['private_right'][where[0],None,:]
     pred.append(x[i]@m.T)
    errs=np.array([nerr(a,b) for a,b in zip(pred,[xx@tt.T for xx,tt in zip(x,targets)])]);aligned=errs[k:];outliers=errs[:k];wall=time.perf_counter()-t0
    if name in ('mirror_rank1_private','native_scalar_phase_private'):private_bytes=int(k*(2*D*4+2))
    elif name=='direct_coeff_rank1_private':private_bytes=int(k*(2*D*4+2))
    elif name=='fedrep_dense_private':private_bytes=int(k*(D*D*4+2))
    else:private_bytes=0
    if name in ('mirror_no_private','mirror_rank1_private','native_scalar_phase_private'):code_bytes=NC*4
    elif name in ('direct_coeff_rank1_private','fedrep_dense_private'):code_bytes=NC*8
    else:code_bytes=0
    viewmac=NC*2*D*D if code_bytes else 0;infermac=NT*2*D*D*NC
    rows.append({'seed':seed,'heterogeneity_rate':rate,'private_clients':k,'method':name,'payload_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'overall_mean_nMSE':float(np.mean(errs)),'aligned_mean_nMSE':float(np.mean(aligned)) if len(aligned) else float('nan'),'outlier_mean_nMSE':float(np.mean(outliers)) if len(outliers) else float('nan'),'private_factor_bytes_raw':private_bytes,'client_code_bytes_total':code_bytes,'client_communication_bytes_total':private_bytes+code_bytes,'examples_evaluated':NC*NT,'optimizer_updates':0,'active_dense_MAC_proxy':infermac,'view_generation_MACs':viewmac,'wall_seconds_serialize_eval':wall})
 out=OUT/('development.csv' if args.dev_only else 'results.csv')
 with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
 for r in rows:print(json.dumps(r,sort_keys=True))
if __name__=='__main__':main()
