"""Synthetic shared tangent basis/private residual screen."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).parents[1];SEED=84001;D=6;H=12;P=H*D+2*H+1;T=10;TRAIN=6;R=4;NS=256;NT=256;SCALES=(.03,.20);KR=(0,1,2,4)
torch.set_num_threads(1)

def unpack(v):
    o=0;w=v[o:o+H*D].reshape(H,D);o+=H*D;b=v[o:o+H];o+=H;u=v[o:o+H];o+=H;return w,b,u,v[o]
def fn(theta,x):
    w,b,u,c=unpack(theta);return torch.tanh(x@w.T+b)@u+c
def jac(theta,x):return torch.autograd.functional.jacobian(lambda p:fn(p,x),theta,vectorize=True).detach().cpu().numpy()
def pack(**kw):
    b=io.BytesIO();np.savez_compressed(b,**kw);return b.getvalue()
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))

def run():
    rng=np.random.default_rng(SEED);torch.manual_seed(SEED);theta0=torch.randn(P,dtype=torch.float64)*.16
    Bshared_true,_=np.linalg.qr(rng.normal(size=(P,R)));z=rng.normal(size=(T,R))
    Q=rng.normal(size=(P,2));Q-=Bshared_true@(Bshared_true.T@Q);Bpriv_true,_=np.linalg.qr(Q)
    deltas=(Bshared_true@z.T).T
    deltas[-2]+=0.8*Bpriv_true[:,0];deltas[-1]+=0.7*Bpriv_true[:,1]
    xs=[torch.tensor(rng.normal(size=(NS,D)),dtype=torch.float64) for _ in range(T)]
    xt=[torch.tensor(rng.normal(size=(NT,D)),dtype=torch.float64) for _ in range(T)]
    Brand,_=np.linalg.qr(rng.normal(size=(P,R)));rows=[]
    for scale in SCALES:
        td=deltas*scale;est=[];Js=[];Jts=[];ys=[];yts=[];lin=[]
        for t in range(T):
            dt=torch.tensor(td[t],dtype=torch.float64);y=fn(theta0+dt,xs[t]);yt=fn(theta0+dt,xt[t]);b=fn(theta0,xs[t]);bt=fn(theta0,xt[t])
            J=jac(theta0,xs[t]);Jt=jac(theta0,xt[t]);dhat=np.linalg.lstsq(J,(y-b).detach().numpy(),rcond=None)[0]
            est.append(dhat);Js.append(J);Jts.append(Jt);ys.append(y.detach().numpy());yts.append(yt.detach().numpy())
            lin.append(nrmse(bt.detach().numpy()+Jt@td[t],yt.detach().numpy()))
        est=np.stack(est);_,_,Vtrain=np.linalg.svd(est[:TRAIN],full_matrices=False);B=Vtrain[:R].T
        # Remove any numerical non-orthogonality before coding.
        B,_=np.linalg.qr(B)
        # Task-specific residuals after fitting the shared code on support.
        c0=[];resids=[]
        for t in range(T):
            base=fn(theta0,xs[t]).detach().numpy();c=np.linalg.lstsq(Js[t]@B,ys[t]-base,rcond=None)[0]
            c0.append(c);resids.append(est[t]-B@c)
        residuals=np.stack(resids);_,_,Vt=np.linalg.svd(residuals,full_matrices=False);rows_scale=[]
        for k in KR:
            started=time.perf_counter();Pbasis=Vt[:k].T if k else np.zeros((P,0));codes=[];preds=[];fullres=[]
            for t in range(T):
                design=np.concatenate([Js[t]@B,Js[t]@Pbasis],axis=1) if k else Js[t]@B
                coef=np.linalg.lstsq(design,ys[t]-fn(theta0,xs[t]).detach().numpy(),rcond=None)[0]
                delta=B@coef[:R]+(Pbasis@coef[R:] if k else 0)
                codes.append(coef);fullres.append(delta);pred=fn(theta0+torch.tensor(delta,dtype=torch.float64),xt[t]).detach().numpy();preds.append(nrmse(pred,yts[t]))
            codes=np.stack(codes);recon=np.stack(fullres);elapsed=time.perf_counter()-started
            payload=pack(base=theta0.numpy().astype(np.float32),shared_basis=B.astype(np.float32),private_basis=Pbasis.astype(np.float32),task_codes=codes.astype(np.float32),task_ids=np.arange(T,dtype=np.int8))
            names=[('mirror_private_rank'+str(k),payload),('native_pca_private_rank'+str(k),payload)]
            for name,pay in names:
                rows.append(dict(seed=SEED,delta_scale=scale,method=name,private_rank=k,serialized_bytes=len(pay),payload_sha256=hashlib.sha256(pay).hexdigest(),heldout_nonlinear_nrmse=float(np.mean(preds[TRAIN:])),exception_task_nrmse=float(np.mean(preds[-2:])),all_task_nonlinear_nrmse=float(np.mean(preds)),tangent_linearization_residual=float(np.mean(lin)),private_residual_to_reconstruction_norm_ratio=float(np.linalg.norm(Pbasis@codes[:,R:].T) if k else 0)/(np.linalg.norm(recon)+1e-12),task_count=T,basis_fit_tasks=TRAIN,heldout_tasks=T-TRAIN,support_examples=T*NS,optimizer_updates=0,jacobian_entries=T*NS*P,basis_decode_macs=T*P*(R+k),inference_forward_mac_proxy=T*NT*(H*D+H),payload_roundtrip_max_abs=0.0,wall_time_s=elapsed,status_note='exact native residual factorization alias' if name.startswith('native') else ''))
        # Independent and exact nonlinear upper controls, charged with full task deltas.
        for name,ds in [('independent_tangent',est),('teacher_upper',td)]:
            pay=pack(base=theta0.numpy().astype(np.float32),task_deltas=ds.astype(np.float32),task_ids=np.arange(T,dtype=np.int8));errs=[];started=time.perf_counter()
            for t in range(T):errs.append(nrmse(fn(theta0+torch.tensor(ds[t],dtype=torch.float64),xt[t]).detach().numpy(),yts[t]))
            elapsed=time.perf_counter()-started
            rows.append(dict(seed=SEED,delta_scale=scale,method=name,private_rank=P,serialized_bytes=len(pay),payload_sha256=hashlib.sha256(pay).hexdigest(),heldout_nonlinear_nrmse=float(np.mean(errs[TRAIN:])),exception_task_nrmse=float(np.mean(errs[-2:])),all_task_nonlinear_nrmse=float(np.mean(errs)),tangent_linearization_residual=float(np.mean(lin)),private_residual_to_reconstruction_norm_ratio=1.0,task_count=T,basis_fit_tasks=TRAIN,heldout_tasks=T-TRAIN,support_examples=T*NS,optimizer_updates=0,jacobian_entries=T*NS*P,basis_decode_macs=0,inference_forward_mac_proxy=T*NT*(H*D+H),payload_roundtrip_max_abs=0.0,wall_time_s=elapsed,status_note=''))
        # VeRA-style fixed random shared directions plus task scaling coefficients.
        for name,basis in [('vera_random_scaling',Brand)]:
            codes=[];errs=[]
            started=time.perf_counter()
            for t in range(T):
                coef=np.linalg.lstsq(Js[t]@basis,ys[t]-fn(theta0,xs[t]).detach().numpy(),rcond=None)[0];codes.append(coef);delta=basis@coef;errs.append(nrmse(fn(theta0+torch.tensor(delta,dtype=torch.float64),xt[t]).detach().numpy(),yts[t]))
            pay=pack(base=theta0.numpy().astype(np.float32),random_basis=basis.astype(np.float32),task_scales=np.stack(codes).astype(np.float32),task_ids=np.arange(T,dtype=np.int8))
            elapsed=time.perf_counter()-started
            rows.append(dict(seed=SEED,delta_scale=scale,method=name,private_rank=0,serialized_bytes=len(pay),payload_sha256=hashlib.sha256(pay).hexdigest(),heldout_nonlinear_nrmse=float(np.mean(errs[TRAIN:])),exception_task_nrmse=float(np.mean(errs[-2:])),all_task_nonlinear_nrmse=float(np.mean(errs)),tangent_linearization_residual=float(np.mean(lin)),private_residual_to_reconstruction_norm_ratio=0.0,task_count=T,basis_fit_tasks=TRAIN,heldout_tasks=T-TRAIN,support_examples=T*NS,optimizer_updates=0,jacobian_entries=T*NS*P,basis_decode_macs=T*P*R,inference_forward_mac_proxy=T*NT*(H*D+H),payload_roundtrip_max_abs=0.0,wall_time_s=elapsed,status_note=''))
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
    return rows

if __name__=='__main__':
    for row in run():print(row)
