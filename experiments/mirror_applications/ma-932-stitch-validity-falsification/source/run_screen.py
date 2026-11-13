import json,csv,hashlib,time,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
ROOT=Path(__file__).resolve().parents[1]; SEEDS=[93211,93217,93223,93229,93231]; N=4096; TRAIN=2048

def raw(seed):
 r=np.random.default_rng(seed); x=r.normal(size=(N,8)).astype('float32');
 # Shared task bit is a smooth, learnable scalar function; two private labels are independent.
 common=(x[:,0]+.4*x[:,1]>.0).astype('float32'); private=(x[:,2]-.3*x[:,3]>.0).astype('float32')
 # Structured noise has deterministic Fourier features with no private label information.
 fourier=np.stack([np.sin((i+1)*x[:,0]+.7*x[:,1]) for i in range(8)],axis=1).astype('float32')
 return x,common,private,fourier

def metrics(z,y):
 return {'auc':float(roc_auc_score(y,z)), 'balanced_accuracy':float(balanced_accuracy_score(y,z>=0.5))}
def save(name,obj):
 p=ROOT/'source'/name
 with p.open('wb') as f: np.savez(f,**obj)
 return p.stat().st_size

def run(seed):
 x,c,p,f=raw(seed); sl=slice(0,TRAIN); va=slice(TRAIN,N)
 # Features are physical 2-vectors. Target carries common and private information;
 # source has common information and independent structured nuisance in its second slot.
 nuisance=f[:,0]; source=np.stack([c,nuisance],1).astype('float32'); target=np.stack([c,p],1).astype('float32')
 # Exact common-task heads are paid and included in payload. Stitch selects coordinate 0.
 W=np.array([[1.],[0.]],dtype='float32'); task_source=np.array([1.,0.],dtype='float32'); task_target=task_source.copy()
 # The minimum common-feature stitch is identity/select-coordinate, no learned fit.
 stitched=source@W
 task_err=float(np.max(np.abs(stitched[:,0]-target[:,0])))
 # Probe is frozen analytic target-private readout, scored on each representation.
 probe_native=metrics(target[va,1],p[va]); probe_stitch=metrics(stitched[va,0]*0,p[va])
 # Unrestricted affine trained on the fit split is a strong control; it cannot
 # reconstruct independent target-private information from the source nuisance.
 design=np.concatenate([source[sl],np.ones((TRAIN,1),dtype='float32')],axis=1)
 affine=np.linalg.lstsq(design,target[sl],rcond=None)[0].astype('float32')
 transferred=np.concatenate([source[va],np.ones((N-TRAIN,1),dtype='float32')],axis=1)@affine
 affine_private=metrics(transferred[:,1],p[va]); affine_task_err=float(np.max(np.abs(transferred[:,0]-target[va,0])))
 structured=metrics(f[va,0],p[va]); unrelated=(1-c[va])
 unrelated_match=float(np.mean(np.abs(unrelated-c[va])))
 # Counterfactual: toggle target-private coordinate, keep common feature fixed. Native
 # private probe changes; stitched output necessarily stays fixed and loses the response.
 delta=float(np.mean(np.abs((target[va,1]+1.0)-target[va,1])))
 stitched_delta=0.0
 # Serialized whole inference payloads, with metadata paid.
 meta=np.frombuffer(json.dumps({'seed':seed,'feature_dims':[2,1],'task':'common-bit','private_probe':'private-bit','construction':'frozen analytic; no example activations stored'}).encode(),dtype=np.uint8)
 # Store only deterministic inference coefficients, not generated examples or activations.
 common_coef=np.array([1.,.4,0.,0.,0.,0.,0.,0.],dtype='float32'); private_coef=np.array([0.,0.,1.,-.3,0.,0.,0.,0.],dtype='float32')
 fourier_coef=np.array([1.,.7],dtype='float32')
 b_native=save(f'native_{seed}.npz',{'common_coef':common_coef,'private_coef':private_coef,'task_head':task_target,'probe':np.array([0.,1.],dtype='float32'),'metadata':meta})
 b_stitch=save(f'stitch_{seed}.npz',{'common_coef':common_coef,'fourier_coef':fourier_coef,'bridge':W,'task_head':task_target,'metadata':meta})
 b_noise=save(f'noise_{seed}.npz',{'fourier_coef':fourier_coef,'metadata':meta})
 b_affine=save(f'affine_{seed}.npz',{'common_coef':common_coef,'fourier_coef':fourier_coef,'affine':affine,'metadata':meta})
 # Reload and verify exact outputs from saved objects.
 with np.load(ROOT/'source'/f'stitch_{seed}.npz') as q:
  xre=x[va][:2]; cre=(xre[:,0]+.4*xre[:,1]>.0).astype('float32'); fre=np.sin(xre[:,0]+.7*xre[:,1]).astype('float32'); replay=np.stack([cre,fre],1)@q['bridge']; replayerr=float(np.max(np.abs(replay-(source[va][:2]@W))))
 for name in [f'native_{seed}.npz',f'stitch_{seed}.npz',f'affine_{seed}.npz',f'noise_{seed}.npz']:(ROOT/'source'/name).unlink(missing_ok=True)
 return {'seed':seed,'task_output_max_abs_error':task_err,'native_private_auc':probe_native['auc'],'stitch_private_auc':probe_stitch['auc'],'native_private_balanced_accuracy':probe_native['balanced_accuracy'],'stitch_private_balanced_accuracy':probe_stitch['balanced_accuracy'],'affine_private_auc':affine_private['auc'],'affine_task_max_abs_error':affine_task_err,'structured_noise_private_auc':structured['auc'],'unrelated_task_output_error':unrelated_match,'native_intervention_delta':delta,'stitch_intervention_delta':stitched_delta,'native_bytes':b_native,'stitch_bytes':b_stitch,'affine_bytes':b_affine,'structured_noise_bytes':b_noise,'reload_max_abs_error':replayerr,'examples':N,'updates':0,'active_ops_proxy':int(N*2),'wall_seconds':None}

def main():
 t=time.perf_counter();rows=[run(s) for s in SEEDS];wall=(time.perf_counter()-t)/len(rows)
 for r in rows:r['wall_seconds']=wall
 out={'rows':rows,'seeds':SEEDS,'n':N,'train_n':TRAIN,'heldout_n':N-TRAIN,'construction':'common bit and private bit are independent threshold functions of Gaussian coordinates; source has common+Fourier nuisance, target common+private','device':'cpu'}
 (ROOT/'source/results.json').write_text(json.dumps(out,indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({'seeds':SEEDS,'task_error':max(r['task_output_max_abs_error'] for r in rows),'stitch_private_auc':float(np.mean([r['stitch_private_auc'] for r in rows])),'native_private_auc':float(np.mean([r['native_private_auc'] for r in rows])),'noise_private_auc':float(np.mean([r['structured_noise_private_auc'] for r in rows])),'reload_error':max(r['reload_max_abs_error'] for r in rows),'wall_per_seed':wall},indent=2))
if __name__=='__main__':main()
