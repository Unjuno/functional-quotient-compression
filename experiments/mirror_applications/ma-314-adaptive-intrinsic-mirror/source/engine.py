"""MA-314 adaptive intrinsic-dimension allocation screen."""
from __future__ import annotations

import hashlib
import io
import time
import zipfile
from pathlib import Path

import numpy as np

D, K, T = 16, 8, 48
NS, NV, NT = 64, 64, 128
LEVELS = (2, 4, 8)
THRESHOLD = 1e-4
RADII = np.array([0.25, 0.20, 0.15, 0.10], dtype=np.float32)
GRID = np.linspace(-np.pi, np.pi, 720, endpoint=False, dtype=np.float32)
METHODS = ("tied", "adaptive_said", "adaptive_coeff", "adaptive_mirror", "fixed_said8", "independent")


def make_world(seed: int):
    rng = np.random.default_rng(seed)
    theta0 = rng.normal(0, 0.15, D).astype(np.float32)
    q, _ = np.linalg.qr(rng.normal(size=(D, K)))
    basis = q.astype(np.float32)
    dims = np.tile(np.array(LEVELS, dtype=np.int32), T // len(LEVELS))
    rng.shuffle(dims)
    z = np.zeros((T, K), dtype=np.float32)
    angles = rng.uniform(-np.pi, np.pi, (T, K // 2)).astype(np.float32)
    for t, d in enumerate(dims):
        for p in range(d // 2):
            z[t, 2*p:2*p+2] = RADII[p] * np.array([np.cos(angles[t, p]), np.sin(angles[t, p])])
    targets = theta0[None, :] + z @ basis.T
    xs, ys = [], []
    for n in (NS, NV, NT):
        x = rng.normal(size=(T, n, D)).astype(np.float32)
        y = np.einsum("tnd,td->tn", x, targets, optimize=True)
        xs.append(x); ys.append(y)
    return theta0, basis, RADII.copy(), dims, targets, *xs, *ys


def normalized_error(x, y, theta):
    pred = x @ theta
    mse = float(np.mean((pred-y)**2))
    return mse, mse / max(float(np.mean(y*y)), 1e-12)


def _fit_one(method, x, y, theta0, basis, radii, t, d):
    p = d // 2
    features = x @ basis[:, :d]
    residual = y - x @ theta0
    if method in ("adaptive_said", "fixed_said8"):
        coeff = np.linalg.lstsq(features, residual, rcond=None)[0].astype(np.float32)
    elif method == "adaptive_coeff":
        coeff = np.linalg.lstsq(features, residual, rcond=None)[0].astype(np.float16).astype(np.float32)
    elif method == "adaptive_mirror":
        # Jointly optimize the active pair angles: fitting each angle against the
        # entire residual would incorrectly omit every other active pair.
        coeff = np.zeros(p, dtype=np.float32)
        cg, sg = np.cos(GRID), np.sin(GRID)
        for _ in range(4):
            for pair in range(p):
                other = np.zeros_like(residual)
                for j in range(p):
                    if j == pair: continue
                    other += radii[j] * (features[:, 2*j]*np.cos(coeff[j]) + features[:, 2*j+1]*np.sin(coeff[j]))
                a = radii[pair] * features[:, 2*pair]
                b = radii[pair] * features[:, 2*pair+1]
                target = residual-other
                errs = np.mean((a[:, None]*cg + b[:, None]*sg - target[:, None])**2, axis=0)
                coeff[pair] = GRID[int(np.argmin(errs))]
        coeff = coeff.astype(np.float16)
    else:
        raise ValueError(method)
    return coeff


def _decode(method, theta0, basis, radii, coeff, d):
    if method == "adaptive_mirror":
        z = np.zeros(K, dtype=np.float32)
        for p, angle in enumerate(coeff):
            z[2*p:2*p+2] = radii[p] * np.array([np.cos(float(angle)), np.sin(float(angle))])
    else:
        z = np.zeros(K, dtype=np.float32)
        z[:d] = np.asarray(coeff, dtype=np.float32)
    return theta0 + basis @ z


def _fit_adaptive(method, theta0, basis, radii, xtr, ytr, xval, yval):
    chosen, dims, candidates = [], np.zeros(T, dtype=np.uint8), []
    fit_ops = 0
    for t in range(T):
        per_d = {}
        for d in LEVELS:
            if method == "fixed_said8" and d != 8:
                continue
            code = _fit_one(method, xtr[t], ytr[t], theta0, basis, radii, t, d)
            theta = _decode(method, theta0, basis, radii, code, d)
            _, val = normalized_error(xval[t], yval[t], theta)
            per_d[d] = (code, val)
            if method == "adaptive_mirror": fit_ops += 4 * NS * len(GRID) * (d//2) * 6
            elif method in ("adaptive_said", "fixed_said8"): fit_ops += NS*d*d + d**3
            else: fit_ops += NS*d*d + d**3
        if method == "fixed_said8": d = 8
        else: d = next((d for d in LEVELS if d in per_d and per_d[d][1] <= THRESHOLD), 8)
        code = per_d[d][0]
        dims[t] = d
        chosen.append(code)
        candidates.append(per_d)
    return {"theta0": theta0, "basis": basis, "radii": radii,
            "dims": dims, "codes": chosen}, fit_ops, candidates


def make_state(method, theta0, basis, radii, xtr, ytr, xval, yval):
    if method == "tied":
        return {"theta0": theta0}, 0, None
    if method == "independent":
        w = np.stack([np.linalg.lstsq(xtr[t], ytr[t], rcond=None)[0] for t in range(T)]).astype(np.float32)
        return {"weights": w}, T * (NS*D*D + D**3), None
    fitted, ops, candidates = _fit_adaptive(method, theta0, basis, radii, xtr, ytr, xval, yval)
    # For all methods charge the common full maximum basis and radii. Codes are packed
    # into one ragged flat array plus one byte of active-dimension metadata per task.
    if method == "adaptive_mirror": flat = np.concatenate(fitted["codes"]).astype(np.float16)
    elif method == "fixed_said8": flat = np.concatenate(fitted["codes"]).astype(np.float32)
    else: flat = np.concatenate(fitted["codes"]).astype(np.float32 if method == "adaptive_said" else np.float16)
    state = {"theta0": theta0, "basis": basis, "radii": radii, "dims": fitted["dims"], "codes": flat}
    return state, ops, candidates


def decode_task(method, state, t):
    if method == "tied": return state["theta0"]
    if method == "independent": return state["weights"][t]
    d = int(state["dims"][t])
    stride = (d//2 if method == "adaptive_mirror" else d)
    offset = sum((int(v)//2 if method == "adaptive_mirror" else int(v)) for v in state["dims"][:t])
    code = state["codes"][offset:offset+stride]
    return _decode(method, state["theta0"], state["basis"], state["radii"], code, d)


def npy_bytes(array):
    b = io.BytesIO()
    np.lib.format.write_array(b, np.ascontiguousarray(array), allow_pickle=False)
    return b.getvalue()


def save_payload(path: Path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in sorted(state):
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            zf.writestr(info, npy_bytes(state[name]))
    return path.stat().st_size


def load_payload(path):
    with zipfile.ZipFile(path) as zf:
        return {name[:-4]: np.load(io.BytesIO(zf.read(name)), allow_pickle=False)
                for name in sorted(zf.namelist())}


def evaluate(method, state, x, y, fit_ops):
    started = time.perf_counter()
    errs, norms, per_task = [], [], []
    for t in range(T):
        mse, nmse = normalized_error(x[t], y[t], decode_task(method, state, t))
        errs.append(mse); norms.append(nmse); per_task.append(nmse)
    wall = time.perf_counter()-started
    examples = T*x.shape[1]
    dims = state.get("dims", np.full(T, 0, dtype=np.uint8))
    if method in ("adaptive_said", "adaptive_coeff", "adaptive_mirror"):
        active_ops = float(np.mean([D*int(d)+D+(6*int(d)//2 if method=="adaptive_mirror" else 0) for d in dims]))
    elif method == "fixed_said8": active_ops = D*K+D
    else: active_ops = D
    return {"mse":float(np.mean(errs)), "normalized_mse":float(np.mean(norms)),
            "max_task_normalized_mse":float(np.max(per_task)), "test_examples":examples,
            "support_examples":T*NS, "validation_examples":T*NV, "optimizer_updates":0,
            "fit_compute_proxy":int(fit_ops), "active_ops_per_example":active_ops,
            "active_ops_total":int(active_ops*examples), "wall_time_s":wall,
            "examples_per_s":examples/max(wall,1e-12), "mean_selected_dimension":float(np.mean(dims)),
            "dimension_accuracy":None}


def run_world(seed, outdir):
    theta0,basis,radii,true_dims,targets,xtr,xval,xte,ytr,yval,yte = make_world(seed)
    rows=[]
    for method in METHODS:
        start=time.perf_counter()
        state,ops,candidates=make_state(method,theta0,basis,radii,xtr,ytr,xval,yval)
        encode=time.perf_counter()-start
        if method not in ("tied","independent"):
            state["theta0" if "theta0" not in state else "theta0"] = state.get("theta0",theta0)
        path=outdir/f"{seed}_{method}.npz"
        size=save_payload(path,state); loaded=load_payload(path)
        metrics=evaluate(method,loaded,xte,yte,ops)
        dimacc=float(np.mean(loaded["dims"]==true_dims)) if "dims" in loaded else 0.0
        metrics["dimension_accuracy"]=dimacc
        # For adaptive methods also retain quality by the hidden teacher dimension.
        grouped={str(d):float(np.mean([metrics_task(method,loaded,xte,yte,t) for t in range(T) if true_dims[t]==d])) for d in LEVELS}
        rows.append({"seed":seed,"method":method,"serialized_bytes":size,
                     "tensor_bytes":sum(v.nbytes for v in state.values()),"support_examples":T*NS,
                     "validation_examples":T*NV,"test_examples":T*NT,"optimizer_updates":0,
                     "fit_compute_proxy":ops,"active_ops_per_example":metrics["active_ops_per_example"],
                     "active_ops_total":metrics["active_ops_total"],"encode_wall_time_s":encode,
                     "wall_time_s":metrics["wall_time_s"],"examples_per_s":metrics["examples_per_s"],
                     "mse":metrics["mse"],"normalized_mse":metrics["normalized_mse"],
                     "max_task_normalized_mse":metrics["max_task_normalized_mse"],
                     "mean_selected_dimension":metrics["mean_selected_dimension"],
                     "dimension_accuracy":dimacc,"quality_by_true_dimension":grouped,
                     "payload_sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    return rows


def metrics_task(method,state,x,y,t):
    return normalized_error(x[t],y[t],decode_task(method,state,t))[1]
