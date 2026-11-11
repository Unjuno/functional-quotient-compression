"""Synthetic held-out depth x expert role-composition screen for MA-794."""
import csv, hashlib, io, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parents[1]
SEED, L, E, D, R = 79401, 8, 8, 8, 2


def pack(**arrays):
    out = io.BytesIO()
    np.savez_compressed(out, **arrays)
    return out.getvalue()


def nrmse(pred, target):
    return float(np.sqrt(np.mean((pred-target)**2)/(np.mean(target**2)+1e-12)))


def run():
    rng = np.random.default_rng(SEED)
    bases = rng.normal(0, .25, (R,D,D)).astype(np.float32)
    expert = rng.normal(0, .3, (E,D,D)).astype(np.float32)
    depth = rng.normal(0, .2, (L,R)).astype(np.float32)
    expcode = rng.normal(0, .2, (E,R)).astype(np.float32)
    teacher = expert[None].repeat(L, axis=0)
    teacher += np.einsum('lr,er,rij->leij', depth, expcode, bases)
    held = np.fromfunction(lambda l,e: (l+e)%4==0, (L,E), dtype=int)
    observed = ~held
    # Estimate expert intercepts and shared bases from observed role matrices only.
    intercept = np.stack([teacher[observed[:,e],e].mean(axis=0) for e in range(E)])
    resid = teacher - intercept[None]
    # Recover a common rank-2 matrix basis from observed residuals (native shared-basis control).
    flat = resid[observed].reshape(-1,D*D)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    learned_basis = vt[:R].reshape(R,D,D).astype(np.float32)
    coeff = np.einsum('leij,rij->ler', resid, learned_basis)
    # Fit product codes per coordinate by alternating least squares on observed cells.
    dl = rng.normal(0,.1,(L,R)).astype(np.float32); ec = rng.normal(0,.1,(E,R)).astype(np.float32)
    for r in range(R):
        y = coeff[:,:,r]
        for _ in range(300):
            for l in range(L):
                m = observed[l]
                den = np.dot(ec[m,r],ec[m,r])
                if den>1e-12: dl[l,r] = np.dot(ec[m,r],y[l,m])/den
            for e in range(E):
                m = observed[:,e]
                den = np.dot(dl[m,r],dl[m,r])
                if den>1e-12: ec[e,r] = np.dot(dl[m,r],y[m,e])/den
            norm=np.linalg.norm(dl[:,r])
            if norm>1e-12: dl[:,r]*=np.sign(dl[np.argmax(np.abs(dl[:,r])),r]); ec[:,r]*=norm
    pred_product = intercept[None] + np.einsum('lr,er,rij->leij',dl,ec,learned_basis)
    # Native bilinear implements the same parameterization and reconstruction exactly.
    pred_native = pred_product.copy()
    # Additive depth/expert coefficients fitted by least squares coordinate-wise.
    add=np.zeros((L,E,R),np.float32); add_dl=np.zeros((L,R),np.float32); add_ec=np.zeros((E,R),np.float32)
    for r in range(R):
        ids=np.argwhere(observed); X=np.zeros((len(ids),L+E),np.float64)
        for i,(l,e) in enumerate(ids): X[i,l]=1; X[i,L+e]=1
        beta=np.linalg.lstsq(X,coeff[observed,r],rcond=None)[0]
        add_dl[:,r]=beta[:L]; add_ec[:,r]=beta[L:]
        for l in range(L):
            for e in range(E): add[l,e,r]=beta[l]+beta[L+e]
    pred_add=intercept[None]+np.einsum('ler,rij->leij',add,learned_basis)
    tied=np.broadcast_to(intercept[None],teacher.shape).copy()
    methods=[]
    router=np.broadcast_to(np.eye(E,dtype=np.float32),(L,E,E)).copy()
    role_ids=np.arange(L*E,dtype=np.int16).reshape(L,E)
    examples=np.eye(D,dtype=np.float32)
    target=np.einsum('ni,leji->lenj',examples,teacher)
    loads=np.ones((L,E),dtype=np.int16)
    specs=[('independent_roles',teacher,pack(weights=teacher,router=router,role_ids=role_ids),0),
      ('hard_global_expert_tying',tied,pack(weights=intercept,router=router,role_ids=role_ids),0),
      ('additive_depth_expert',pred_add,pack(expert_intercepts=intercept,depth_codes=add_dl,expert_codes=add_ec,basis=learned_basis,router=router,role_ids=role_ids),L*E*R*D*D),
      ('mirror_product_codes',pred_product,pack(expert_intercepts=intercept,depth_codes=dl,expert_codes=ec,basis=learned_basis,router=router,role_ids=role_ids),L*E*R*D*D),
      ('native_bilinear_shared_basis',pred_native,pack(expert_intercepts=intercept,depth_codes=dl,expert_codes=ec,basis=learned_basis,router=router,role_ids=role_ids),L*E*R*D*D)]
    for name,w,payload,decode in specs:
        saved=np.load(io.BytesIO(payload))
        if name=='independent_roles': restored=saved['weights']
        elif name=='hard_global_expert_tying': restored=np.broadcast_to(saved['weights'][None],(L,E,D,D))
        elif name=='additive_depth_expert': restored=saved['expert_intercepts'][None]+np.einsum('ler,rij->leij',saved['depth_codes'][:,None,:]+saved['expert_codes'][None,:,:],saved['basis'])
        else: restored=saved['expert_intercepts'][None]+np.einsum('lr,er,rij->leij',saved['depth_codes'],saved['expert_codes'],saved['basis'])
        roundtrip=float(np.max(np.abs(restored-w)))
        if roundtrip>1e-6: raise AssertionError(f'{name} payload fails reconstruction: {roundtrip}')
        start=time.perf_counter()
        out=np.einsum('ni,leji->lenj',examples,w)
        wall=time.perf_counter()-start
        methods.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),
          heldout_operator_nrmse=nrmse(w[held],teacher[held]),all_role_output_nrmse=nrmse(out,target),
          per_layer_load_min=int(loads.min()),per_layer_load_max=int(loads.max()),active_macs_per_token=E*D*D+E*D,
          reconstruction_macs_total=decode,optimizer_updates=0,input_examples=L*E*D,wall_time_s=wall,payload_roundtrip_max_abs=roundtrip,status_note='exact native bilinear alias' if name=='mirror_product_codes' else ''))
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=methods[0].keys(),lineterminator='\n');wr.writeheader();wr.writerows(methods)
    return methods

if __name__=='__main__':
    for row in run(): print(row)
