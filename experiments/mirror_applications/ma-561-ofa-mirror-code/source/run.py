import argparse, csv, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

TRAIN=[(6,1,1),(12,1,3),(6,3,3),(12,3,1)]
HELD=[(6,1,3),(12,1,1),(6,3,1),(12,3,3)]
ALL=TRAIN+HELD
D=12

def cf(c):
    w,d,k=c
    return np.array([(w-9)/3,(d-2)/1,(k-2)/1],dtype=np.float64)

def teacher_coeff(): return np.array([0.31,-0.22,0.17])
def teacher_angle(c): return cf(c)@teacher_coeff()

def givens(h,theta):
    z=h.copy(); a=z[...,0::2].copy(); b=z[...,1::2].copy()
    co=np.cos(theta); si=np.sin(theta)
    z[...,0::2]=co*a-si*b; z[...,1::2]=si*a+co*b
    return z

def base(x,c,weights):
    w,depth,kernel=c; A,B,C=weights
    h=np.tanh(x@A[:,:w])
    # Kernel code controls a deterministic local mixing strength.
    mix=0.72 if kernel==1 else 0.96
    for _ in range(depth): h=np.tanh(mix*h@B[:w,:w]+(1-mix)*h)
    return h@C[:w,:]

def hidden(x,c,weights):
    w,depth,kernel=c; A,B,C=weights
    h=np.tanh(x@A[:,:w]); mix=0.72 if kernel==1 else 0.96
    for _ in range(depth): h=np.tanh(mix*h@B[:w,:w]+(1-mix)*h)
    return h

def target(x,c,weights):
    w=c[0]; _,_,C=weights
    return givens(hidden(x,c,weights),teacher_angle(c))@C[:w,:]

def normalized_mse(pred,y): return float(np.mean((pred-y)**2)/(np.mean(y*y)+1e-20))

def archive(path,**kw):
    path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(path,**kw); return path.stat().st_size

def run(seed,out,split):
    rng=np.random.default_rng(seed)
    A=rng.normal(0,.36,(D,D)); B=np.eye(D)+rng.normal(0,.10,(D,D)); C=rng.normal(0,.30,(D,D))
    weights=(A,B,C); X={c:rng.normal(size=(384,D)) for c in ALL}
    Y={c:target(X[c],c,weights) for c in ALL}
    # Fit only the three Mirror angle-law coefficients on the four exposed configurations.
    def mirror_res(q): return np.concatenate([(givens(hidden(X[c],c,weights),q@cf(c))@C[:c[0],:]-Y[c]).ravel() for c in TRAIN])
    t=time.perf_counter(); opt=least_squares(mirror_res,np.zeros(3),max_nfev=80); fit_seconds=time.perf_counter()-t; q=opt.x
    # Generic rank-one output residual: delta y = (x.u) * v * (a . config); fit jointly on training configurations.
    # Fix the scale gauge by normalizing u after every objective evaluation.
    def unpack(z):
        u=z[:D]; nu=np.linalg.norm(u)+1e-12; u=u/nu; v=z[D:2*D]*nu; a=z[2*D:]
        return u,v,a
    def rank_res(z):
        u,v,a=unpack(z); parts=[]
        for c in TRAIN:
            delta=Y[c]-base(X[c],c,weights)
            pred=np.outer(X[c]@u, v*(a@cf(c)))
            parts.append((pred-delta).ravel())
        return np.concatenate(parts)
    z0=np.r_[rng.normal(size=D),rng.normal(scale=.1,size=D),np.zeros(3)]
    t=time.perf_counter(); rankfit=least_squares(rank_res,z0,max_nfev=120); rank_seconds=time.perf_counter()-t
    u,v,a=unpack(rankfit.x)
    # Full independent configuration maps: least-squares linear maps per configuration, including held-out.
    # They are an oracle-calibrated storage/quality ceiling, not a learnable transfer baseline.
    full={c:(givens(np.eye(c[0]),teacher_angle(c))@C[:c[0],:]) for c in ALL}
    payload=out/split/str(seed); payload.mkdir(parents=True,exist_ok=True)
    shared=payload/'shared.npz'; shared_b=archive(shared,A=A,B=B,C=C)
    mirror_b=archive(payload/'mirror.npz',q=q)
    rank_b=archive(payload/'rank1.npz',u=u,v=v,a=a)
    full_b=archive(payload/'independent.npz',**{f'M{i}':full[c] for i,c in enumerate(ALL)})
    rows=[]
    for c in ALL:
        if c in TRAIN: continue
        x=X[c]; y=Y[c]
        pm=givens(hidden(x,c,weights),q@cf(c))@C[:c[0],:]
        pr=base(x,c,weights)+np.outer(x@u,v*(a@cf(c)))
        preds={'native':base(x,c,weights),'mirror':pm,'rank1':pr,'independent':hidden(x,c,weights)@full[c]}
        for method,pred in preds.items():
            rows.append({'seed':seed,'split':split,'width':c[0],'depth':c[1],'kernel':c[2],'method':method,'nrmse':np.sqrt(normalized_mse(pred,y)),'payload_bytes':shared_b+{'native':0,'mirror':mirror_b,'rank1':rank_b,'independent':full_b}[method],'shared_bytes':shared_b,'method_bytes':{'native':0,'mirror':mirror_b,'rank1':rank_b,'independent':full_b}[method],'fit_seconds':fit_seconds if method=='mirror' else rank_seconds if method=='rank1' else 0.0,'forward_ms_384':None})
    # Benchmark inference on a single heldout config, 30 batches; CPU timing only.
    c=HELD[0]; x=X[c]
    funcs={'native':lambda:base(x,c,weights),'mirror':lambda:givens(hidden(x,c,weights),q@cf(c))@C[:c[0],:],'rank1':lambda:base(x,c,weights)+np.outer(x@u,v*(a@cf(c))),'independent':lambda:hidden(x,c,weights)@full[c]}
    for method,fn in funcs.items():
        t=time.perf_counter()
        for _ in range(30): fn()
        ms=(time.perf_counter()-t)*1000/30
        for row in rows:
            if row['method']==method: row['forward_ms_384']=ms
    p=payload/'metrics.json'; p.write_text(json.dumps({'seed':seed,'split':split,'mirror_coeff':q.tolist(),'rank1_u':u.tolist(),'rank1_v':v.tolist(),'rank1_config':a.tolist(),'bytes':{'shared':shared_b,'mirror':mirror_b,'rank1':rank_b,'independent':full_b},'optimizer_updates':0,'fit_seconds':{'mirror':fit_seconds,'rank1':rank_seconds}},indent=2)+'\n')
    with (payload/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--split',choices=['dev','fresh'],required=True); ap.add_argument('--seeds',type=int,nargs='+',required=True); args=ap.parse_args()
    rows=[]
    for seed in args.seeds: rows+=run(seed,args.out,args.split)
    target=args.out/args.split/'RESULTS_CORE.csv'; target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
if __name__=='__main__': main()
