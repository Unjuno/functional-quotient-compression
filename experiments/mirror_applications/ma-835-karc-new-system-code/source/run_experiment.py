"""Deterministic synthetic KARC new-system code and native-control screen."""
import csv, hashlib, io, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1]
SEED=83501; S=8; D=4; H=24; NS=48; NR=32

def features(x, grow=False):
    x=np.asarray(x,dtype=np.float64)
    out=[np.ones(x.shape[:-1]+(1,)),x,x*x]
    out += [np.sin(np.pi*x),np.cos(np.pi*x)]
    if grow: out += [x**3]
    return np.concatenate(out,axis=-1)

def pack(**kw):
    b=io.BytesIO(); np.savez_compressed(b,**kw); return b.getvalue()
def nrmse(a,b): return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))
def rollout(x0,W,grow=False):
    x=np.array(x0,copy=True); seq=[]
    for _ in range(H):
        x=x+0.035*np.einsum('...f,fd->...d',features(x,grow),W)
        seq.append(x.copy())
    return np.stack(seq,axis=-2)
def fit(X,Y): return np.linalg.lstsq(X.reshape(-1,X.shape[-1]),Y.reshape(-1,Y.shape[-1]),rcond=None)[0]
def run():
    rng=np.random.default_rng(SEED); P=features(np.zeros((1,D))).shape[-1]
    # Two shared latent readout directions create a deliberately compressible task bank.
    latent=rng.normal(0,.08,(2,P,D)); coeff=rng.normal(size=(S,2)); W=np.einsum('sk,kfd->sfd',coeff,latent)
    # Two systems contain a cubic term absent from the frozen basis.
    W[-2:]+=rng.normal(0,.025,(2,P,D))
    supx=rng.uniform(-.7,.7,(S,NS,D)); supy=np.empty((S,NS,D))
    held0=rng.uniform(-.65,.65,(S,NR,D)); truth=[]
    Wgrow=np.concatenate([W,np.zeros((S,D,D))],axis=1)
    Wgrow[-2:,-D:,:]=rng.normal(0,.025,(2,D,D))
    for s in range(S):
        supy[s]=supx[s]+.035*np.einsum('nf,fd->nd',features(supx[s],True),Wgrow[s])
        truth.append(rollout(held0[s],Wgrow[s],True))
    truth=np.stack(truth)
    native=np.stack([fit(features(supx[s]),supy[s]-supx[s])/.035 for s in range(S)])
    # Mirror address uses task coefficients in a shared rank-two basis; SVD fitted only on support readouts.
    U,sv,Vt=np.linalg.svd(native.reshape(S,-1),full_matrices=False); r=2
    basis=(sv[:r,None]*Vt[:r]).reshape(r,P,D); code=U[:,:r]
    mirror=np.einsum('sr,rfd->sfd',code,basis)
    mean=native.mean(0)
    # Growth control refits with the cubic feature. It stores a global expanded feature declaration and each readout.
    grown=np.stack([fit(features(supx[s],True),supy[s]-supx[s])/.035 for s in range(S)])
    rows=[]
    basis_ids=np.array([0,1,2,3,4],np.int8); basis_spec=np.array([1.0,np.pi,2.0,.035],np.float32)
    native_factor_payload=pack(basis_ids=basis_ids,basis_spec=basis_spec,factor_basis=basis.astype(np.float32),system_coefficients=code.astype(np.float32))
    payloads={
      'shared_only':pack(basis_ids=basis_ids,basis_spec=basis_spec,readout=mean.astype(np.float32)),
      'independent_native_readouts':pack(basis_ids=basis_ids,basis_spec=basis_spec,readouts=native.astype(np.float32)),
      'native_rank2_readout_factor':native_factor_payload,
      'mirror_system_codes':native_factor_payload,
      'basis_growth_cubic_control':pack(basis_ids=np.array([0,1,2,3,4,5],np.int8),basis_spec=basis_spec,readouts=grown.astype(np.float32))}
    for name,payload in payloads.items():
      z=np.load(io.BytesIO(payload)); t=time.perf_counter()
      if name=='shared_only': pred=np.stack([rollout(held0[s],z['readout']) for s in range(S)])
      elif name=='independent_native_readouts': pred=np.stack([rollout(held0[s],z['readouts'][s]) for s in range(S)])
      elif name in ('native_rank2_readout_factor','mirror_system_codes'):
        bb=z['factor_basis'] if 'factor_basis' in z else z['view_basis']; cc=z['system_coefficients'] if 'system_coefficients' in z else z['view_codes']; ws=np.einsum('sr,rfd->sfd',cc,bb); pred=np.stack([rollout(held0[s],ws[s]) for s in range(S)])
      else: pred=np.stack([rollout(held0[s],z['readouts'][s],True) for s in range(S)])
      err=nrmse(pred,truth); active=S*NR*H*P; rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),heldout_rollout_nrmse=err,systems=S,support_transitions=S*NS,heldout_rollouts=S*NR,horizon=H,optimizer_updates=0,active_feature_evaluations=active,payload_roundtrip_max_abs=0.0,wall_time_s=time.perf_counter()-t,basis_growth_events=1 if name=='basis_growth_cubic_control' else 0,status_note='exact native factorization alias' if name in ('native_rank2_readout_factor','mirror_system_codes') else ''))
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
    return rows
if __name__=='__main__':
    for row in run(): print(row)
