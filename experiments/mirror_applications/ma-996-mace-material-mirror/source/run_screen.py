#!/usr/bin/env python3
"""CPU synthetic conservative radial-potential screen for MA-996.
No MACE code, weights, or real material data are used.
"""
import argparse, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CENTERS = np.array([0.9, 1.4, 1.9, 2.4, 2.9], dtype=np.float64)
PAIR_TYPES = [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
D = len(PAIR_TYPES) * len(CENTERS)
GLOBAL_SEED = 99600

def generator():
    rng=np.random.default_rng(GLOBAL_SEED)
    theta=rng.normal(0.0,0.08,D)
    u=rng.normal(size=(D,2))
    u,_=np.linalg.qr(u)
    u=u*0.06
    return theta,u

def pair_id(a,b):
    if a>b: a,b=b,a
    return PAIR_TYPES.index((a,b))

def config(rng,n=5):
    # Rejection sampling avoids singular pair directions while keeping a varied geometry.
    pts=[]
    while len(pts)<n:
        x=rng.uniform(0,3.5,3)
        if all(np.linalg.norm(x-y)>0.65 for y in pts): pts.append(x)
    species=rng.integers(0,3,n)
    return np.asarray(pts),species

def design(pos,sp,centers=CENTERS,width=0.45):
    n=len(sp); k=len(centers); x=np.zeros((1+3*n,D),dtype=np.float64)
    for i in range(n):
      for j in range(i+1,n):
        delta=pos[i]-pos[j]; r=np.linalg.norm(delta); p=pair_id(int(sp[i]),int(sp[j]))
        phi=np.exp(-0.5*((r-centers)/width)**2)
        dphi=phi*(-(r-centers)/(width*width))
        x[0,p*k:(p+1)*k]+=phi
        # Force = -gradient of pair energy.
        unit=delta/r
        for q in range(k):
          f=-dphi[q]*unit
          x[1+3*i:1+3*i+3,p*k+q]+=f
          x[1+3*j:1+3*j+3,p*k+q]-=f
    return x

def dataset(rng,coeff,count,width):
    xs=[]; ys=[]
    for _ in range(count):
        p,s=config(rng)
        x=design(p,s,width=width)
        xs.append(x); ys.append(x@coeff)
    return np.vstack(xs),np.concatenate(ys)

def fit(x,y): return np.linalg.lstsq(x,y,rcond=1e-10)[0]
def nrmse(p,y): return float(np.sqrt(np.mean((p-y)**2))/(np.std(y)+1e-12))
def measure(coeff,x,y):
    pred=x@coeff
    # First row in each block is energy, rest are forces.
    blocks=x.shape[0]//16
    yy=y.reshape(blocks,16); pp=pred.reshape(blocks,16)
    e=nrmse(pp[:,0],yy[:,0]); f=nrmse(pp[:,1:],yy[:,1:])
    return e,f,float(np.sqrt((e*e+f*f)/2))
def rot_matrix(rng):
    q,_=np.linalg.qr(rng.normal(size=(3,3)))
    if np.linalg.det(q)<0: q[:,0]*=-1
    return q

def force_consistency(coeff,rng,width):
    pos,sp=config(rng)
    x=design(pos,sp,width=width); analytic=(x@coeff)[1:].reshape(-1,3)
    eps=1e-6; num=np.zeros_like(pos)
    for i in range(len(pos)):
      for d in range(3):
        plus=pos.copy(); minus=pos.copy(); plus[i,d]+=eps; minus[i,d]-=eps
        ep=(design(plus,sp,width=width)@coeff)[0]
        em=(design(minus,sp,width=width)@coeff)[0]
        num[i,d]=-(ep-em)/(2*eps)
    fd=float(np.max(np.abs(analytic-num)))
    q=rot_matrix(rng); xr=design(pos@q.T,sp,width=width); fr=(xr@coeff)[1:].reshape(-1,3)
    equiv=float(np.max(np.abs(fr-analytic@q.T)))
    return fd,equiv

def learn_dev(seed,width):
    theta,u=generator(); rng=np.random.default_rng(seed)
    cs=[]
    for _ in range(12):
      z=rng.normal(size=2); c=theta+u@z
      x,y=dataset(rng,c,20,width); cs.append(fit(x,y))
    cs=np.stack(cs); mean=cs.mean(axis=0); _,_,vt=np.linalg.svd(cs-mean,full_matrices=False)
    basis=vt[:2].T
    # Resolve fresh targets on held-out configurations after eight calibration configurations.
    vals=[]
    for _ in range(4):
      z=rng.normal(size=2); c=theta+u@z
      xc,yc=dataset(rng,c,8,width); xv,yv=dataset(rng,c,20,width)
      m=fit(xc@basis,yc); vals.append(measure(mean+basis@m,xv,yv)[2])
    return mean,basis,float(np.mean(vals))

def main():
 p=argparse.ArgumentParser(); p.add_argument('mode',choices=['dev','fresh','verify']); p.add_argument('--world',type=int); a=p.parse_args()
 out=ROOT/'runs'; out.mkdir(exist_ok=True)
 if a.mode=='dev':
   rows=[]
   for width in [0.35,0.45,0.55,0.65]:
      _,_,score=learn_dev(99601,width); rows.append({'width':width,'combined_nrmse':score})
   chosen=min(rows,key=lambda r:r['combined_nrmse'])
   (out/'development.json').write_text(json.dumps({'seed':99601,'candidates':rows,'selected_width':chosen['width'],'rank':2},indent=2)+'\n')
   print(json.dumps({'development':rows,'selected':chosen},indent=2))
 elif a.mode=='fresh':
   if a.world not in [99611,99613,99617]: raise SystemExit('fresh world must be preregistered')
   cfg=json.loads((out/'development.json').read_text()); width=cfg['selected_width']
   theta,u=generator(); rngdev=np.random.default_rng(99601); devcs=[]
   for _ in range(12):
      z=rngdev.normal(size=2); c=theta+u@z; x,y=dataset(rngdev,c,20,width); devcs.append(fit(x,y))
   devcs=np.stack(devcs); base=devcs.mean(0); _,_,vt=np.linalg.svd(devcs-base,full_matrices=False); b=vt[:2].T
   rng=np.random.default_rng(a.world); target_codes=rng.normal(size=(4,2)); targets=[theta+u@z for z in target_codes]
   methods={k:[] for k in ['shared_only','scalar_gate','mirror_rank2','native_rank2','sparse_update','independent_fit']}
   metrics=[]; m_codes=[]; independent=[]; sparse_codes=[]; scalar_alphas=[]; wall_start=time.perf_counter()
   # Development-selected 6-coefficient support.
   support=np.argsort(np.mean(np.abs(devcs-base),axis=0))[-6:]
   for idx,c in enumerate(targets):
      xc,yc=dataset(rng,c,8,width); xv,yv=dataset(rng,c,24,width)
      # Mirror and native low-rank are deliberately same parameterization; bytes/hash expose alias.
      m=fit(xc@b,yc-xc@base); cm=base+b@m; m_codes.append(m)
      alpha=float(np.dot(xc@base,yc)/(np.dot(xc@base,xc@base)+1e-12)); cs=alpha*base; scalar_alphas.append(alpha)
      xs=xc[:,support]; ds=fit(xs,yc-xc@base); csp=base.copy(); csp[support]+=ds; sparse_codes.append(ds)
      ci=fit(xc,yc); independent.append(ci)
      preds={'shared_only':base,'scalar_gate':cs,'mirror_rank2':cm,'native_rank2':cm.copy(),'sparse_update':csp,'independent_fit':ci}
      for name,cc in preds.items(): methods[name].append(measure(cc,xv,yv))
      fd,eq=force_consistency(cm,np.random.default_rng(a.world+idx+100),width)
      metrics.append({'target':idx,'force_gradient_max_abs':fd,'rotation_equivariance_max_abs':eq,'calibration_examples':8,'heldout_examples':24})
   # Store fully charged inference payloads, including basis/centers/width and per-target data.
   states={
    'shared_only':{'centers':CENTERS,'width':np.array(width),'base':base},
    'scalar_gate':{'centers':CENTERS,'width':np.array(width),'base':base,'alpha':np.asarray(scalar_alphas)},
    'mirror_rank2':{'centers':CENTERS,'width':np.array(width),'base':base,'basis':b,'codes':np.stack(m_codes)},
    'native_rank2':{'centers':CENTERS,'width':np.array(width),'base':base,'basis':b,'codes':np.stack(m_codes)},
    'sparse_update':{'centers':CENTERS,'width':np.array(width),'base':base,'indices':support.astype(np.int16),'residuals':np.stack(sparse_codes)},
    'independent_fit':{'centers':CENTERS,'width':np.array(width),'coefficients':np.stack(independent)}}
   bytes_by={}
   for name,state in states.items():
      path=out/f'{a.world}_{name}.npz'; np.savez(path,**state); bytes_by[name]=path.stat().st_size
   # Re-open every file and verify arrays round-trip exactly.
   roundtrip={}
   for name,state in states.items():
      loaded=np.load(out/f'{a.world}_{name}.npz'); roundtrip[name]=all(np.array_equal(loaded[k],v) for k,v in state.items())
   summary={}
   for name,vals in methods.items():
      arr=np.asarray(vals); summary[name]={'energy_nrmse_mean':float(arr[:,0].mean()),'force_nrmse_mean':float(arr[:,1].mean()),'combined_nrmse_mean':float(arr[:,2].mean()),'serialized_bytes':bytes_by[name],'inference_mac_proxy_per_config':int(2*10*len(CENTERS)+(48 if name in ['mirror_rank2','native_rank2'] else 6 if name=='sparse_update' else D if name=='scalar_gate' else 0)),'roundtrip_exact':roundtrip[name]}
   result={'world':a.world,'width':width,'rank':2,'targets':4,'calibration_examples_per_target':8,'heldout_examples_per_target':24,'methods':summary,'integrity_checks':metrics,'units':'synthetic eV-style coefficient units','wall_time_seconds':float(time.perf_counter()-wall_start),'train_examples_total':12*20,'calibration_examples_total':4*8,'optimizer_updates':0}
   path=out/f'fresh_{a.world}.json'; path.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
 else:
   files=list(out.glob('fresh_*.json')); good=[]
   for f in files:
     d=json.loads(f.read_text()); good.append(all(v['roundtrip_exact'] for v in d['methods'].values()) and max(x['force_gradient_max_abs'] for x in d['integrity_checks'])<1e-7 and max(x['rotation_equivariance_max_abs'] for x in d['integrity_checks'])<1e-10)
   print(json.dumps({'fresh_files':[str(f) for f in files],'checks_passed':good,'all_pass':bool(good) and all(good)},indent=2))
if __name__=='__main__': main()
