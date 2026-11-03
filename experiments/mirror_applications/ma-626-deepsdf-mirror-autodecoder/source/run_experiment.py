import argparse,csv,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1]
def sdf(x,c,r): return np.linalg.norm(x-c,axis=-1)-r
def fit(x,y): return least_squares(lambda z:sdf(x,z[:2],z[2])-y,np.array([0.,0.,1.]),max_nfev=200)
def nrmse(y,z): return float(np.sqrt(np.mean((y-z)**2))/max(np.sqrt(np.mean(y*y)),1e-12))
def bilinear(grid,x,lo=-2.,hi=2.):
 n=grid.shape[0]; u=np.clip((x-lo)/(hi-lo)*(n-1),0,n-1-1e-9); i=np.floor(u).astype(int); a=u-i; x0=i[:,0];y0=i[:,1];ax=a[:,0];ay=a[:,1]
 return ((1-ax)*(1-ay)*grid[x0,y0]+ax*(1-ay)*grid[x0+1,y0]+(1-ax)*ay*grid[x0,y0+1]+ax*ay*grid[x0+1,y0+1])
def run(seed):
 rng=np.random.default_rng(seed); nshape=16; codes=[]; supports=[]; audits=[]; truths=[]; grids=[]; fit_evals=[]
 centers=rng.uniform(-.55,.55,(nshape,2)); radii=rng.uniform(.45,.95,nshape)
 for k in range(nshape):
  xs=rng.uniform(-1.5,1.5,(64,2)); xa=rng.uniform(-1.5,1.5,(256,2)); y=sdf(xs,centers[k],radii[k]); fitr=fit(xs,y)
  codes.append(fitr.x); supports.append(xs); audits.append(xa); truths.append(sdf(xa,centers[k],radii[k])); fit_evals.append(fitr.nfev)
  axis=np.linspace(-2,2,64); xx,yy=np.meshgrid(axis,axis,indexing='ij'); points=np.stack([xx,yy],axis=-1); grids.append(sdf(points,centers[k],radii[k]).astype(np.float32))
 codes=np.asarray(codes); grids=np.asarray(grids,dtype=np.float32); indices=np.arange(nshape,dtype=np.uint8); schema=np.frombuffer(b'SDF2D-CIRCLE-V1',dtype=np.uint8)
 common={'codes':codes,'shape_indices':indices,'decoder_schema':schema,'domain':np.array([-2.,2.])}
 payloads={'mirror_shared_decoder':common,'native_direct_shape_conditioner':common,
           'independent_sdf_grids':{'grids':grids,'shape_indices':indices,'grid_bounds':np.array([-2.,2.,-2.,2.]),'grid_schema':np.frombuffer(b'SDF2D-GRID64-LINEAR',dtype=np.uint8)}}
 rows=[]
 for method,arrays in payloads.items():
  path=ROOT/'source'/f'{seed}_{method}.npz'; np.savez_compressed(path,**arrays); nbytes=path.stat().st_size
  for k in range(nshape):
   truth=truths[k]
   if method=='independent_sdf_grids': pred=bilinear(grids[k],audits[k])
   else: pred=sdf(audits[k],codes[k,:2],codes[k,2])
   err=nrmse(truth,pred)
   # Approximate scalar arithmetic operation counts per point, plus support fit evaluations.
   ops_point={'mirror_shared_decoder':11,'native_direct_shape_conditioner':5,'independent_sdf_grids':12}[method]
   rows.append({'seed':seed,'shape_id':k,'method':method,'serialized_bytes':nbytes,'heldout_nrmse':err,'support_points':64,'fit_evaluations':fit_evals[k],'active_compute_proxy':int(256*ops_point+fit_evals[k]*64*5),'optimizer_updates':0})
 return rows
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,required=True);args=ap.parse_args();start=time.perf_counter();rows=[]
 for seed in args.seeds:rows+=run(seed)
 elapsed=time.perf_counter()-start
 for r in rows:r['wall_time_s']=elapsed
 (ROOT/'source'/'development.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note'])
  for r in rows:w.writerow(['translated_scaled_circles',r['seed'],r['method'],r['serialized_bytes'],r['support_points'],0,r['active_compute_proxy'],r['wall_time_s'],'audit_nrmse',r['heldout_nrmse'],'fit_evaluations',r['fit_evaluations'],'development; fresh sealed'])
 print(json.dumps(rows,indent=2))
