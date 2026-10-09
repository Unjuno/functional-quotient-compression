#!/usr/bin/env python3
"""MA-553 synthetic FiLM task-layer factorization audit."""
import argparse,csv,io,json,time,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];TASKS=6;LAYERS=4;C=32;SEEDS=(55301,55302)
TRAIN=tuple((t,l) for t in range(TASKS) for l in range(LAYERS) if not (t>=4 and l in (1,2,3)))
HOLD=tuple((t,l) for t in range(4,6) for l in (1,2,3))
def pack(a):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',compression=zipfile.ZIP_STORED) as z:
  for k in sorted(a):
   q=io.BytesIO();np.save(q,a[k],allow_pickle=False);z.writestr(zipfile.ZipInfo(k+'.npy',(1980,1,1,0,0,0)),q.getvalue())
 return b.getvalue()
def run(seed):
 rng=np.random.default_rng(seed); T=rng.normal(size=(TASKS,4));L=rng.normal(size=(LAYERS,4));B=rng.normal(size=(4,2*C))*.2
 # Target affine states have shared task/layer latent factorization plus mild noise.
 Y=np.empty((TASKS,LAYERS,2*C),np.float32)
 for t in range(TASKS):
  for l in range(LAYERS):Y[t,l]=((T[t]*L[l])@B + .15*rng.normal(size=2*C)).astype(np.float32)
 # Oracle structured coordinate fit for m=(task code, layer code, shared basis); known teacher latent, fit B from observed pairs.
 X=np.array([np.outer(T[t],L[l]).reshape(-1) for t,l in TRAIN]);Z=np.stack([Y[t,l] for t,l in TRAIN]); coef=np.linalg.lstsq(X,Z,rcond=None)[0].reshape(4,4,2*C)
 pred=np.empty_like(Y)
 for t in range(TASKS):
  for l in range(LAYERS):pred[t,l]=np.outer(T[t],L[l]).reshape(-1)@coef.reshape(16,2*C)
 # Native HyperFormer-like additive generator: task embedding + layer embedding, least-squares fit all channels.
 H=np.array([np.eye(TASKS+LAYERS)[t].tolist()+np.eye(TASKS+LAYERS)[TASKS+l].tolist() for t,l in TRAIN])
 # Simpler actual direct-FiLM table and factorized code payloads.
 independent={'affine':Y.astype(np.float32),'meta':np.frombuffer(b'FiLM-independent-24pairs',dtype=np.uint8)}
 factor={'task_code':T.astype(np.float32),'layer_code':L.astype(np.float32),'basis':coef.astype(np.float32),'meta':np.frombuffer(b'factorized-affine-code',dtype=np.uint8)}
 mirror_bytes=len(pack(factor));direct_bytes=mirror_bytes
 flat_bytes=len(pack({'affine':Y.astype(np.float32),'meta':np.frombuffer(b'flat-direct-table',dtype=np.uint8)}))
 # Small shared linear hypernetwork on one-hot task/layer addresses fitted on visible pairs.
 Xh=np.array([np.eye(TASKS+LAYERS)[t].tolist()+np.eye(TASKS+LAYERS)[TASKS+l].tolist() for t,l in TRAIN])
 # Actually concatenate embedding indices as one-hot of 10 coords.
 Xh=np.array([np.eye(TASKS+LAYERS)[t]+np.eye(TASKS+LAYERS)[TASKS+l] for t,l in TRAIN])
 hc=np.linalg.lstsq(Xh,Z,rcond=None)[0]
 hp={}
 # predict task/layer one-hot affine outputs from shared linear generator
 for t in range(TASKS):
  for l in range(LAYERS): hp[t,l]=(np.eye(TASKS+LAYERS)[t]+np.eye(TASKS+LAYERS)[TASKS+l])@hc
 hyper_bytes=len(pack({'linear_generator':hc.astype(np.float32),'meta':np.frombuffer(b'hyperformer-linear-control',dtype=np.uint8)}))
 # evaluate functional output on locked Gaussian probes
 probes=rng.normal(size=(512,C)).astype(np.float32); rows=[]
 for method in ('no_modulation','independent_film','flat_table','direct_factorized','mirror_factorized','hyperformer_linear'):
  se=[];oe=[];outs=[]
  for t,l in HOLD:
   truth=Y[t,l]; gt,bt=truth[:C],truth[C:]
   if method=='no_modulation':q=np.zeros(2*C,np.float32)
   elif method=='independent_film' or method=='flat_table':q=Y[t,l]
   elif method in ('direct_factorized','mirror_factorized'):q=pred[t,l]
   else:q=hp[t,l]
   se.append(np.mean((q-truth)**2));out=probes*q[:C]+q[C:];ref=probes*gt+bt;oe.append(np.mean((out-ref)**2)/max(np.mean(ref**2),1e-12));outs.append(np.round(out,5).tobytes())
  b={'no_modulation':64,'independent_film':len(pack(independent)),'flat_table':flat_bytes,'direct_factorized':direct_bytes,'mirror_factorized':mirror_bytes,'hyperformer_linear':hyper_bytes}[method]
  t0=time.perf_counter();_=[probes@np.diag(np.ones(C)) for _ in HOLD];wall=time.perf_counter()-t0
  rows.append({'world':seed,'method':method,'serialized_bytes':b,'train_examples':len(TRAIN),'optimizer_updates':0,'active_compute_proxy':len(HOLD)*512*C*2,'wall_time_s':f'{wall:.6f}','heldout_gamma_beta_nmse':f'{np.mean(se):.9g}','heldout_output_nmse':f'{np.mean(oe):.9g}','distinct_functions':len(set(outs)),'status_note':'synthetic known rank-4 interaction; oracle least squares'})
 return rows,{'direct':direct_bytes,'mirror':mirror_bytes,'flat':flat_bytes,'independent':len(pack(independent)),'hyperformer':hyper_bytes}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(ROOT/'artifacts'/'dev_results.csv'));a=ap.parse_args();rows=[];sizes=[]
 for seed in SEEDS:r,s=run(seed);rows+=r;sizes.append(s)
 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 with open(a.out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 print(json.dumps({'rows':len(rows),'sizes':sizes},indent=2))
if __name__=='__main__':main()
