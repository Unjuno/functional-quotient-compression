"""MA-332: audit exact gauge views versus function-changing hidden rotation."""
from __future__ import annotations
import argparse, csv, hashlib, json, struct, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IN_DIM, HIDDEN, OUT_DIM, N = 16, 24, 8, 4096
DEV = [33221, 33222]
FRESH = [33231, 33232, 33233]
METHODS = ["baseline", "permutation_view", "positive_scale_compensated", "negative_scale_compensated", "givens_view_uncompensated", "givens_view_compensated", "independent_full"]

def arrays(world):
    rng = np.random.default_rng(world)
    x = rng.standard_normal((N, IN_DIM), dtype=np.float32)
    w1 = (rng.standard_normal((IN_DIM, HIDDEN), dtype=np.float32) / np.sqrt(IN_DIM)).astype(np.float32)
    b1 = rng.standard_normal(HIDDEN, dtype=np.float32) * .1
    w2 = (rng.standard_normal((HIDDEN, OUT_DIM), dtype=np.float32) / np.sqrt(HIDDEN)).astype(np.float32)
    b2 = rng.standard_normal(OUT_DIM, dtype=np.float32) * .1
    p = rng.permutation(HIDDEN).astype(np.int16)
    scale = np.exp(rng.uniform(-.8, .8, HIDDEN)).astype(np.float32)
    negscale = scale.copy(); negscale[::2] *= -1
    return x, w1, b1, w2, b2, p, scale, negscale

def givens(h, theta=.73):
    y = h.copy()
    c, s = np.float32(np.cos(theta)), np.float32(np.sin(theta))
    y[..., 0] = c*h[...,0] - s*h[...,1]
    y[..., 1] = s*h[...,0] + c*h[...,1]
    return y

def evaluate(method, x, w1, b1, w2, b2, p, scale, negscale):
    h = np.maximum(x @ w1 + b1, 0)
    if method in ("baseline", "independent_full"):
        return h @ w2 + b2
    if method == "permutation_view":
        return h[:, p] @ w2[p] + b2
    if method == "positive_scale_compensated":
        hp = np.maximum((x @ w1 + b1) * scale, 0)
        return hp @ (w2 / scale[:,None]) + b2
    if method == "negative_scale_compensated":
        hp = np.maximum((x @ w1 + b1) * negscale, 0)
        return hp @ (w2 / negscale[:,None]) + b2
    if method == "givens_view_uncompensated":
        return givens(h) @ w2 + b2
    if method == "givens_view_compensated":
        return givens(h) @ (givens(np.eye(HIDDEN, dtype=np.float32), -0.73) @ w2) + b2
    raise ValueError(method)

def payload(method, w1, b1, w2, b2, p, scale, negscale):
    meta = {"method": method, "shapes": [[*a.shape] for a in (w1,b1,w2,b2)], "dtype":"float32", "version":1}
    chunks = [w1.tobytes(), b1.tobytes(), w2.tobytes(), b2.tobytes()]
    if method == "permutation_view":
        meta["code"] = "int16 hidden permutation"; chunks.append(p.tobytes())
    elif method == "positive_scale_compensated":
        meta["code"] = "float32 positive hidden scales"; chunks.append(scale.tobytes())
    elif method == "negative_scale_compensated":
        meta["code"] = "float32 signed hidden scales"; chunks.append(negscale.tobytes())
    elif method.startswith("givens_view"):
        meta["code"] = "float32 angle"; chunks.append(struct.pack("<f", .73))
    raw = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()
    return b"MA332\0" + struct.pack("<I", len(raw)) + raw + b"".join(chunks)

def load_payload(blob):
    assert blob[:6] == b"MA332\0"
    n = struct.unpack("<I", blob[6:10])[0]
    meta = json.loads(blob[10:10+n])
    pos = 10+n; base=[]
    for shape in meta["shapes"]:
        count=int(np.prod(shape)); size=count*4
        base.append(np.frombuffer(blob[pos:pos+size], dtype=np.float32).copy().reshape(shape)); pos+=size
    code=None
    if meta["method"] == "permutation_view":
        code=np.frombuffer(blob[pos:],dtype=np.int16).copy()
    elif meta["method"] in ("positive_scale_compensated","negative_scale_compensated"):
        code=np.frombuffer(blob[pos:],dtype=np.float32).copy()
    elif meta["method"].startswith("givens_view"):
        code=struct.unpack("<f",blob[pos:pos+4])[0]
    return meta, base, code

def nrmse(y, ref):
    return float(np.linalg.norm(y-ref) / max(np.linalg.norm(ref), 1e-12))

def run(phase):
    rows=[]
    for world in (DEV if phase == "development" else FRESH):
        x,w1,b1,w2,b2,p,scale,negscale=arrays(world)
        ref=evaluate("baseline",x,w1,b1,w2,b2,p,scale,negscale)
        for method in METHODS:
            t=time.perf_counter(); out=evaluate(method,x,w1,b1,w2,b2,p,scale,negscale); elapsed=time.perf_counter()-t
            blob=payload(method,w1,b1,w2,b2,p,scale,negscale)
            diff=out-ref
            rows.append({"phase":phase,"world":world,"method":method,"serialized_bytes":len(blob),"examples":N,"optimizer_updates":0,"active_compute_proxy":N*IN_DIM*HIDDEN + N*HIDDEN*OUT_DIM,"wall_time_s":elapsed,"output_nrmse":nrmse(out,ref),"max_abs_difference":float(np.max(np.abs(diff))),"payload_sha256":hashlib.sha256(blob).hexdigest()})
    path=ROOT/f"{phase.upper()}_RESULTS.csv"
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(json.dumps({"phase":phase,"rows":len(rows),"worlds":DEV if phase=="development" else FRESH}))

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--phase",choices=["development","fresh"],required=True);run(p.parse_args().phase)
