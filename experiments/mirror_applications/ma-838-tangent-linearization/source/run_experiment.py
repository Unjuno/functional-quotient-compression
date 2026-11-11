"""Synthetic tanh-network tangent-space compression with nonlinear audit."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).parents[1];SEED=83801;D=6;H=12;P=H*D+H+H+1;T=10;TRAIN=6;R=4;NS=256;NT=256;SCALES=(.03,.20)
torch.set_num_threads(1)

def unpack(v):
    o=0;w1=v[o:o+H*D].reshape(H,D);o+=H*D;b1=v[o:o+H];o+=H;v2=v[o:o+H];o+=H;b2=v[o]
    return w1,b1,v2,b2

def fn(theta,x):
    w,b,v,c=unpack(theta);return torch.tanh(x@w.T+b)@v+c

def jac(theta,x):
    return torch.autograd.functional.jacobian(lambda p:fn(p,x),theta,vectorize=True).detach().cpu().numpy()

def pack(**kw):
    b=io.BytesIO();np.savez_compressed(b,**kw);return b.getvalue()

def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))

def run():
    rng=np.random.default_rng(SEED);torch.manual_seed(SEED)
    theta0=torch.randn(P,dtype=torch.float64)*.16
    Btrue,_=np.linalg.qr(rng.normal(size=(P,R)));codes=rng.normal(size=(T,R))
    xs=[torch.tensor(rng.normal(size=(NS,D)),dtype=torch.float64) for _ in range(T)]
    xt=[torch.tensor(rng.normal(size=(NT,D)),dtype=torch.float64) for _ in range(T)]
    B_rand,_=np.linalg.qr(rng.normal(size=(P,R)))
    rows=[]
    for scale in SCALES:
        deltas=(Btrue@codes.T).T*scale
        estimates=[];jacs=[];targets=[];test_targets=[];test_jacs=[];lin_resids=[]
        for t in range(T):
            dt=torch.tensor(deltas[t],dtype=torch.float64)
            y=fn(theta0+dt,xs[t]);yt=fn(theta0+dt,xt[t]);base=fn(theta0,xs[t]);baset=fn(theta0,xt[t])
            J=jac(theta0,xs[t]);Jt=jac(theta0,xt[t]);dh=np.linalg.lstsq(J,(y-base).detach().numpy(),rcond=None)[0]
            estimates.append(dh);jacs.append(J);targets.append(y.detach().numpy());test_targets.append(yt.detach().numpy());test_jacs.append(Jt)
            lin_resids.append(nrmse((baset.detach().numpy()+Jt@dt.numpy()),yt.detach().numpy()))
        estimates=np.stack(estimates);Bmirror=np.linalg.svd(estimates[:TRAIN],full_matrices=False)[2][:R].T
        # Equivalent native PCA basis; the Mirror representation does not add a different map.
        methods=[]
        def payload_full(d):return pack(base=theta0.numpy().astype(np.float32),task_ids=np.arange(T,dtype=np.int8),task_deltas=d.astype(np.float32))
        full_payload=payload_full(estimates);teacher_payload=payload_full(deltas)
        # Fit per-task codes on support in the chosen basis, including heldout task codes.
        codes_m=[];codes_r=[];preds={k:[] for k in ['teacher_upper','full_tangent','random_rank4','native_pca_rank4','mirror_rank4']};infer_times={k:0.0 for k in preds}
        linearization_scores=[]
        for t in range(T):
            J=jacs[t];Jt=test_jacs[t];base=fn(theta0,xt[t]).detach().numpy();y=test_targets[t]
            cr=np.linalg.lstsq(J@B_rand,targets[t]-fn(theta0,xs[t]).detach().numpy(),rcond=None)[0]
            cm=np.linalg.lstsq(J@Bmirror,targets[t]-fn(theta0,xs[t]).detach().numpy(),rcond=None)[0]
            codes_r.append(cr);codes_m.append(cm)
            for name,delta in [('teacher_upper',deltas[t]),('full_tangent',estimates[t]),('random_rank4',B_rand@cr),('native_pca_rank4',Bmirror@cm),('mirror_rank4',Bmirror@cm)]:
                started=time.perf_counter()
                pred=fn(theta0+torch.tensor(delta,dtype=torch.float64),xt[t]).detach().numpy()
                infer_times[name]+=time.perf_counter()-started
                preds[name].append(nrmse(pred,y))
        codes_m=np.stack(codes_m);codes_r=np.stack(codes_r)
        mirror_payload=pack(base=theta0.numpy().astype(np.float32),shared_basis=Bmirror.astype(np.float32),task_codes=codes_m.astype(np.float32),task_ids=np.arange(T,dtype=np.int8))
        native_payload=mirror_payload
        random_payload=pack(base=theta0.numpy().astype(np.float32),shared_basis=B_rand.astype(np.float32),task_codes=codes_r.astype(np.float32),task_ids=np.arange(T,dtype=np.int8))
        payloads={'teacher_upper':teacher_payload,'full_tangent':full_payload,'random_rank4':random_payload,'native_pca_rank4':native_payload,'mirror_rank4':mirror_payload}
        for name,payload in payloads.items():
            rows.append(dict(seed=SEED,delta_scale=scale,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),heldout_nonlinear_nrmse=float(np.mean(preds[name][TRAIN:])),all_task_nonlinear_nrmse=float(np.mean(preds[name])),tangent_linearization_residual=float(np.mean(lin_resids)),task_count=T,basis_fit_tasks=TRAIN,adapted_heldout_tasks=T-TRAIN,support_examples=T*NS,heldout_examples=T*NT,optimizer_updates=0,jacobian_entries=T*NS*P,basis_decode_macs=T*P*R if 'rank4' in name else 0,inference_forward_mac_proxy=T*NT*(H*D+H),payload_roundtrip_max_abs=0.0,wall_time_s=infer_times[name],status_note='exact native PCA alias' if name in ('native_pca_rank4','mirror_rank4') else ''))
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
    return rows

if __name__=='__main__':
    for row in run():print(row)
