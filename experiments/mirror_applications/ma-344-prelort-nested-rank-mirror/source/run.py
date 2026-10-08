#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"artifacts";SEEDS=(34401,34402,34411,34412,34413);D=16;MAXR=4;NC=64;NT=512

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

def world(seed):
 rng=np.random.default_rng(seed);W0=rng.normal(size=(D,D)).astype(np.float32)*.08;z=rng.normal(size=(D,MAXR));B,_=np.linalg.qr(z);B=B.astype(np.float32);A1=(rng.normal(size=(MAXR,D))*.12).astype(np.float32);A2=(rng.normal(size=(MAXR,D))*.12).astype(np.float32)
 ranks=np.array([1]*24+[2]*24+[4]*16,np.uint8);rng.shuffle(ranks);phase=rng.uniform(-np.pi,np.pi,size=(NC,MAXR)).astype(np.float32);xev=[rng.normal(size=(NT,D)).astype(np.float32) for _ in range(NC)]
 mats=[]
 for i,r in enumerate(ranks):
  a=np.cos(phase[i,:r,None])*A1[:r]+np.sin(phase[i,:r,None])*A2[:r];mats.append(W0+B[:,:r]@a)
 Boff,_=np.linalg.qr(rng.normal(size=(D,MAXR)));Boff=Boff.astype(np.float32);Aoff=(rng.normal(size=(MAXR,D))*.2).astype(np.float32);xoff=rng.normal(size=(NT,D)).astype(np.float32);Moff=W0+Boff@Aoff
 truth=[x@m.T for x,m in zip(xev,mats)];truth_off=xoff@Moff.T
 return W0,B,A1,A2,ranks,phase,mats,xev,truth,Boff,Aoff,xoff,truth_off

def nerr(a,b):return float(np.mean((a-b)**2)/(np.mean(b*b)+1e-30))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dev-only',action='store_true');args=ap.parse_args();OUT.mkdir(exist_ok=True);rows=[]
 for seed in SEEDS[:2] if args.dev_only else SEEDS:
  W0,B,A1,A2,ranks,phase,mats,xev,truth,Boff,Aoff,xoff,t_off=world(seed);ptr=np.r_[0,np.cumsum(ranks.astype(int))].astype(np.uint16);phflat=np.concatenate([phase[i,:ranks[i]] for i in range(NC)]).astype(np.float32);coeff=np.stack([np.cos(phflat),np.sin(phflat)],1).astype(np.float32)
  apre=np.concatenate([np.cos(phase[i,:ranks[i],None])*A1[:ranks[i]]+np.sin(phase[i,:ranks[i],None])*A2[:ranks[i]] for i in range(NC)],axis=0).astype(np.float32)
  rankcode=ranks.astype(np.uint8)
  hard=np.mean(mats,axis=0).astype(np.float32)
  cases={
   'independent_full_models':({'models':np.stack(mats+[W0+Boff@Aoff])},{'kind':'independent_full','count':65}),
   'prelort_active_A_factors':({'base':W0,'B':B,'A_flat':apre,'ptr':ptr,'ranks':rankcode,'private_B':Boff,'private_A':Aoff},{'kind':'prelort_prefix_sharedB','private':True}),
   'mirror_segment_phase_private':({'base':W0,'B':B,'A1':A1,'A2':A2,'phase':phflat,'ptr':ptr,'ranks':rankcode,'private_B':Boff,'private_A':Aoff},{'kind':'segment_phase','private':True}),
   'mirror_segment_phase_no_private':({'base':W0,'B':B,'A1':A1,'A2':A2,'phase':phflat,'ptr':ptr,'ranks':rankcode},{'kind':'segment_phase','private':False}),
   'direct_segment_coefficients_private':({'base':W0,'B':B,'A1':A1,'A2':A2,'coefficients':coeff,'ptr':ptr,'ranks':rankcode,'private_B':Boff,'private_A':Aoff},{'kind':'direct_segment_coefficients','private':True}),
   'native_scalar_phase_private':({'base':W0,'B':B,'A1':A1,'A2':A2,'phase':phflat,'ptr':ptr,'ranks':rankcode,'private_B':Boff,'private_A':Aoff},{'kind':'segment_phase','private':True}),
   'hard_shared_adapter':({'weight':hard},{'kind':'hard_shared'})}
  for name,(arrays,meta) in cases.items():
   tick=time.perf_counter();payload,ld,md=pack(arrays,meta);pred=[]
   if name=='independent_full_models':
    for i,x in enumerate(xev):pred.append(x@ld['models'][i].T)
    offpred=xoff@ld['models'][-1].T
   elif name=='hard_shared_adapter':
    pred=[x@ld['weight'].T for x in xev];offpred=xoff@ld['weight'].T
   else:
    for i,x in enumerate(xev):
     lo,hi=map(int,ld['ptr'][i:i+2]);r=hi-lo
     if name=='prelort_active_A_factors':a=ld['A_flat'][lo:hi];bi=ld['B'][:,:r]
     elif 'coefficients' in ld:a=ld['coefficients'][lo:hi,0,None]*ld['A1'][:r]+ld['coefficients'][lo:hi,1,None]*ld['A2'][:r];bi=ld['B'][:,:r]
     else:a=np.cos(ld['phase'][lo:hi,None])*ld['A1'][:r]+np.sin(ld['phase'][lo:hi,None])*ld['A2'][:r];bi=ld['B'][:,:r]
     pred.append(x@ld['base'].T+(x@a.T)@bi.T)
    if 'private_A' in ld:offpred=xoff@ld['base'].T+(xoff@ld['private_A'].T)@ld['private_B'].T
    else:offpred=xoff@ld['base'].T
   err=[nerr(a,b) for a,b in zip(pred,truth)];perrank={str(int(r)):float(np.mean([e for e,rr in zip(err,ranks) if rr==r])) for r in (1,2,4)};comm={str(r):int((r*4+1 if 'phase' in name or 'mirror' in name or 'native_' in name else r*8+1 if 'coefficients' in name else r*D*4+1)) for r in (1,2,4)}
   active=int(np.sum(ranks));extra=active*2*D if 'phase' in name or 'coefficients' in name or 'mirror' in name else 0;flop=int(NT*2*D*active)
   rows.append({'seed':seed,'method':name,'payload_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'rank1_mean_nMSE':perrank['1'],'rank2_mean_nMSE':perrank['2'],'rank4_mean_nMSE':perrank['4'],'offorbit_nMSE':nerr(offpred,t_off),'active_segments_total':active,'client_code_bytes_by_rank':json.dumps(comm,sort_keys=True),'communication_bytes_total':int(sum(int(r)* (4 if 'phase' in name or 'mirror' in name or 'native_' in name else 8 if 'coefficients' in name else D*4)+1 for r in ranks)),'examples_evaluated':NT*(NC+1),'optimizer_updates':0,'active_inference_MACs':flop,'view_generation_MACs':extra,'wall_seconds_serialize_eval':time.perf_counter()-tick})
 out=OUT/('development.csv' if args.dev_only else 'results.csv')
 with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
 for r in rows:print(json.dumps(r,sort_keys=True))
if __name__=='__main__':main()
