"""MA-268: IA3 diagonal views versus a one-angle activation Mirror."""
import argparse, csv, hashlib, json, math, struct, time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DIN, H, DOUT, TASKS = 16, 32, 8, 8
DEV, FRESH, SEEDS = [26800, 26801], [26810, 26811, 26812], [0, 1, 2]
torch.set_num_threads(2)


def make_world(world, seed):
    g = torch.Generator().manual_seed(world * 100003 + seed * 7919 + 268)
    w1 = torch.randn(DIN, H, generator=g) / math.sqrt(DIN)
    w2 = torch.randn(H, DOUT, generator=g) / math.sqrt(H)
    # Keep activations away from a ReLU boundary under the small rotations.
    def samples(n, offset):
        q = torch.Generator().manual_seed(world * 100003 + seed * 7919 + offset)
        x = torch.randn(n, DIN, generator=q)
        return x, F.relu(x @ w1 + 1.0)
    (xs, hs) = samples(128, 17)
    (xv, hv) = samples(512, 31)
    angles = torch.linspace(-0.7, 0.7, TASKS)
    scales = torch.exp(torch.linspace(-0.7, 0.7, H))[None, :].repeat(TASKS, 1)
    # Independent diagonal task codes vary by world and seed.
    scales = scales * torch.exp(0.08 * torch.randn(TASKS, H, generator=g))
    return w1, w2, xs, hs, xv, hv, angles, scales


def rotate(h, theta):
    c, s = torch.cos(theta), torch.sin(theta)
    y = h.clone()
    y[..., 0] = c * h[..., 0] - s * h[..., 1]
    y[..., 1] = s * h[..., 0] + c * h[..., 1]
    return y


def fit_ia3(h, target, init=None):
    v = nn.Parameter(torch.ones(H) if init is None else init.clone())
    opt = torch.optim.LBFGS([v], lr=0.8, max_iter=100, tolerance_grad=1e-10,
                            tolerance_change=1e-12, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); loss = ((h * v) @ W2_CURRENT - target).square().mean(); loss.backward(); return loss
    t = time.perf_counter(); opt.step(closure)
    return v.detach(), time.perf_counter() - t


def fit_angle(h, target):
    # Deterministic grid + local golden-section search avoids optimizer instability.
    t0 = time.perf_counter()
    grid = torch.linspace(-1.5, 1.5, 601)
    losses = torch.stack([((rotate(h, a) @ W2_CURRENT - target) ** 2).mean() for a in grid])
    idx = int(losses.argmin()); lo = float(grid[max(0, idx - 1)]); hi = float(grid[min(len(grid)-1, idx + 1)])
    phi = (math.sqrt(5) - 1) / 2
    f = lambda a: float(((rotate(h, torch.tensor(a)) @ W2_CURRENT - target) ** 2).mean())
    c, d = hi - phi * (hi - lo), lo + phi * (hi - lo); fc, fd = f(c), f(d)
    for _ in range(48):
        if fc < fd: hi, d, fd = d, c, fc; c = hi - phi * (hi - lo); fc = f(c)
        else: lo, c, fc = c, d, fd; d = lo + phi * (hi - lo); fd = f(d)
    a = (lo + hi) / 2
    return a, time.perf_counter() - t0


def fit_dense(h, target):
    # Linear least squares upper control, one task-specific HxH matrix.
    t = time.perf_counter()
    # Solve (h @ M) @ W2 ~= target; minimum-norm activation-space solution.
    z = target @ torch.linalg.pinv(W2_CURRENT)
    m = torch.linalg.lstsq(h, z).solution
    return m, time.perf_counter() - t


def pack(method, shared, codes):
    # Canonical flat float32 payload plus explicit JSON metadata; all methods use this serializer.
    flat = torch.cat([shared.contiguous().float().view(-1), codes.contiguous().float().view(-1)]).numpy().tobytes()
    meta = json.dumps({"method": method, "shared_shape": list(shared.shape), "code_shape": list(codes.shape), "dtype": "float32"}, sort_keys=True, separators=(",", ":")).encode()
    return b"MA268\0" + struct.pack("<I", len(meta)) + meta + flat


def nrmse(pred, target):
    return float(torch.linalg.vector_norm(pred-target) / torch.linalg.vector_norm(target).clamp_min(1e-12))


def run(phase):
    global W2_CURRENT
    rows = []; worlds = DEV if phase == "development" else FRESH
    for world in worlds:
      for seed in SEEDS:
        w1,w2,xs,hs,xv,hv,angles,scales = make_world(world,seed); W2_CURRENT=w2
        shared=torch.cat([w1.flatten(),w2.flatten()])
        for stratum in ("aligned_rotation", "independent_diagonal"):
          for task in range(TASKS):
            if stratum == "aligned_rotation":
                teacher_code=angles[task]; ht=rotate(hs,teacher_code); hv_t=rotate(hv,teacher_code)
            else:
                teacher_code=scales[task]; ht=hs*teacher_code; hv_t=hv*teacher_code
            target=ht@w2; audit=hv_t@w2
            methods=[]
            methods.append(("no_view", torch.empty(0), torch.zeros_like(audit), 0.0))
            if stratum == "aligned_rotation":
                angle, sec=fit_angle(hs,target); methods.append(("mirror_givens",torch.tensor(angle),rotate(hv,torch.tensor(angle))@w2,sec))
            v,sec=fit_ia3(hs,target); methods.append(("ia3",v,(hv*v)@w2,sec))
            m,sec=fit_dense(hs,target); methods.append(("dense_upper",m,(hv@m)@w2,sec))
            for method,code,pred,fit_sec in methods:
                if method=="no_view": code=torch.empty(0)
                blob=pack(method,shared,code)
                path=ROOT/"artifacts"/"payloads"/f"{world}_{seed}_{stratum}_{task}_{method}.bin";path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob)
                t=time.perf_counter()
                if method=="mirror_givens": _=rotate(hv,code)@w2
                elif method=="ia3": _=(hv*code)@w2
                elif method=="dense_upper": _=(hv@code)@w2
                else: _=hv@w2
                apply_sec=time.perf_counter()-t
                rows.append({"world":world,"seed":seed,"stratum":stratum,"task":task,"method":method,"nrmse":nrmse(pred,audit),"payload_bytes":len(blob),"sha256":hashlib.sha256(blob).hexdigest(),"path":str(path.relative_to(ROOT.parents[2])),"fit_seconds":fit_sec,"apply_seconds":apply_sec,"updates":0 if method in ("no_view","dense_upper","mirror_givens") else 100,"support_examples":128,"audit_examples":512})
    outfile=ROOT/f"{phase.upper()}_RESULTS.csv"
    with outfile.open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=rows[0].keys());wr.writeheader();wr.writerows(rows)
    print(json.dumps({"phase":phase,"rows":len(rows),"output":str(outfile)}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--phase",choices=["development","fresh"],required=True);run(p.parse_args().phase)
