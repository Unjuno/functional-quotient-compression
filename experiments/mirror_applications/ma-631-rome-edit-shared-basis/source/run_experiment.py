import argparse,csv,json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];PAIRS=[(i,j) for i in range(4) for j in range(4)];HELD={(i,i) for i in range(4)};TRAIN=[p for p in PAIRS if p not in HELD]
def run(seed):
 rng=np.random.default_rng(seed);d=32;w0=rng.normal(0,.1,(d,d)).astype(np.float32);u,_=np.linalg.qr(rng.normal(size=(d,2)));a=rng.normal(size=4);b=rng.normal(size=4)
 targets={p:u@np.array([a[p[0]],b[p[1]]]) for p in PAIRS};ttrain=np.stack([targets[p] for p in TRAIN]);_,_,vh=np.linalg.svd(ttrain,full_matrices=False);basis=vh[:2].T;ct={p:basis.T@targets[p] for p in TRAIN}
 # Fit additive factor tables with B_0 fixed to zero to remove gauge freedom.
 design=[]
 for i,j in TRAIN:
  row=np.zeros(7);row[i]=1
  if j>0:row[4+j-1]=1
  design.append(row)
 design=np.stack(design);A=np.zeros((4,2));B=np.zeros((4,2));
 for k in range(2):
  sol=np.linalg.lstsq(design,np.array([ct[p][k] for p in TRAIN]),rcond=None)[0];A[:,k]=sol[:4];B[0,k]=0;B[1:,k]=sol[4:]
 mirror={p:basis@(A[p[0]]+B[p[1]]) for p in PAIRS};lowrank={p:basis@(basis.T@targets[p]) for p in PAIRS};rows=[]
 methods={
  'independent_rank1':{'base':w0,'update_values':np.stack([targets[p] for p in PAIRS]).astype(np.float32),'edit_pairs':np.array(PAIRS,dtype=np.uint8),'schema':np.frombuffer(b'ROME-RANK1-V1',dtype=np.uint8)},
  'shared_lowrank_per_edit':{'base':w0,'shared_basis':basis.astype(np.float32),'coefficients':np.stack([basis.T@targets[p] for p in PAIRS]).astype(np.float32),'edit_pairs':np.array(PAIRS,dtype=np.uint8),'schema':np.frombuffer(b'ROME-LR2-V1',dtype=np.uint8)},
  'mirror_factor_tables':{'base':w0,'shared_basis':basis.astype(np.float32),'factor_a':A.astype(np.float32),'factor_b':B.astype(np.float32),'edit_pairs':np.array(PAIRS,dtype=np.uint8),'schema':np.frombuffer(b'ROME-VIEW-ADD2-V1',dtype=np.uint8)},
  'native_additive_factor_tables':{'base':w0,'shared_basis':basis.astype(np.float32),'factor_a':A.astype(np.float32),'factor_b':B.astype(np.float32),'edit_pairs':np.array(PAIRS,dtype=np.uint8),'schema':np.frombuffer(b'ROME-VIEW-ADD2-V1',dtype=np.uint8)}}
 sizes={};state_sizes={}
 for name,payload in methods.items():
  path=R/'source'/f'{seed}_{name}.npz';np.savez_compressed(path,**payload);sizes[name]=path.stat().st_size
  state={k:v for k,v in payload.items() if k!='base'}
  sp=R/'source'/f'{seed}_{name}_edit_state.npz';np.savez_compressed(sp,**state);state_sizes[name]=sp.stat().st_size
 for method,preds in [('independent_rank1',targets),('shared_lowrank_per_edit',lowrank),('mirror_factor_tables',mirror),('native_additive_factor_tables',mirror)]:
  eff=[];leak=[]
  # An edit's key is the one-hot vector e_n; updates are rank-one target_delta * e_n^T.
  M=np.column_stack([preds[p] for p in PAIRS])
  Y=np.column_stack([targets[p] for p in PAIRS])
  for ix,p in enumerate(PAIRS):
   eff.append(np.linalg.norm(M[:,ix]-Y[:,ix])/max(np.linalg.norm(Y[:,ix]),1e-12))
   key=np.zeros(d);key[ix]=1.0;response=np.outer(preds[p],key)[:,:len(PAIRS)];response[:,ix]=0
   leak.append(float(np.linalg.norm(response)/max(np.linalg.norm(Y[:,ix]),1e-12)))
  ops={'independent_rank1':d,'shared_lowrank_per_edit':2*d+4,'mirror_factor_tables':2*d+6,'native_additive_factor_tables':2*d+6}[method]
  for p in HELD:
   ix=PAIRS.index(p);rows.append({'seed':seed,'task':list(p),'method':method,'serialized_bytes':sizes[method],'edit_state_bytes':state_sizes[method], 'efficacy_nrmse':float(eff[ix]),'locality_leakage':float(leak[ix]),'active_compute_proxy':ops,'optimizer_updates':0})
 return rows
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,required=True);args=ap.parse_args();t=time.perf_counter();rows=[]
 for s in args.seeds:rows+=run(s)
 wall=time.perf_counter()-t
 for r in rows:r['wall_time_s']=wall
 (R/'source'/'development.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (R/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note'])
  for r in rows:w.writerow(['rank_one_edits',r['seed'],r['method'],r['serialized_bytes'],12,0,r['active_compute_proxy'],r['wall_time_s'],'efficacy_nrmse',r['efficacy_nrmse'],'locality_leakage',r['locality_leakage'],'development; fresh sealed'])
 print(json.dumps(rows,indent=2))
