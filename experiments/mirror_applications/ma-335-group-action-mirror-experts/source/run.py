#!/usr/bin/env python3
"""Deterministic synthetic C4 expert-view mechanism and payload audit."""
import argparse, csv, hashlib, io, json, math, time, zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
SEEDS = (33501, 33502, 33520, 33521, 33522)
N = 4096

def rotations():
    return [np.linalg.matrix_power(np.array([[0., -1.], [1., 0.]], np.float32), k) for k in range(4)]

def pack_payload(arrays, meta):
    """Fixed metadata and deterministic zip entries; returned size is authoritative."""
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in sorted(arrays):
            raw = io.BytesIO(); np.save(raw, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", (1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            z.writestr(info, raw.getvalue())
        info = zipfile.ZipInfo("metadata.json", (1980, 1, 1, 0, 0, 0)); info.compress_type=zipfile.ZIP_DEFLATED
        info.external_attr=0o600 << 16
        z.writestr(info, json.dumps(meta, sort_keys=True, separators=(",", ":")).encode())
    payload = bio.getvalue()
    # Force a real reload from the serialized inference state.
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        loaded = {k[:-4]: np.load(io.BytesIO(z.read(k)), allow_pickle=False) for k in z.namelist() if k.endswith(".npy")}
        md = json.loads(z.read("metadata.json"))
    return payload, loaded, md

def make_world(seed):
    rng=np.random.default_rng(seed)
    W=rng.normal(size=(2,2)).astype(np.float32)
    # ensure substantial non-equivariant component
    W += np.array([[1.1, .35], [-.2, -.7]], dtype=np.float32)
    R=rotations()
    related=[r@W@r.T for r in R]
    unrelated=(rng.normal(size=(2,2))*0.8 + np.array([[.4,-.9],[.7,.2]])).astype(np.float32)
    targets=related+[unrelated]
    x=rng.normal(size=(N,2)).astype(np.float32)
    return W, R, targets, x

def mse(A,B,x):
    y=x@B.T; pred=x@A.T
    return float(np.mean((pred-y)**2)/(np.mean(y*y)+1e-30))

def main():
    OUT.mkdir(exist_ok=True)
    rows=[]
    ap=argparse.ArgumentParser(); ap.add_argument("--dev-only",action="store_true"); args=ap.parse_args()
    seeds=SEEDS[:2] if args.dev_only else SEEDS
    for seed in seeds:
        W,R,targets,x=make_world(seed)
        # Exact C4-equivariant projection: Reynolds average commutes with all rotations.
        Weq=np.mean([r@W@r.T for r in R],axis=0)
        cases={
          "independent": ({f"expert_{i}":t for i,t in enumerate(targets)}, list(range(5)), {"kind":"independent","views":5}, 5*8),
          "hard_tied": ({"expert":W}, [0]*5, {"kind":"tied","views":5}, 4),
          "equivariant_projection": ({"expert":Weq}, [0]*5, {"kind":"equivariant","views":5}, 4),
          "group_mirror": ({"base":W,"private":targets[4],"codes":np.array([0,1,2,3],np.uint8)}, [0,1,2,3,4], {"kind":"group_action","group":"C4","views":4,"private_fifth":True}, 4*4+4),
          "group_mirror_no_private": ({"base":W,"codes":np.array([0,1,2,3],np.uint8)}, [0,1,2,3,4], {"kind":"group_action","group":"C4","views":4,"private_fifth":False}, 4*4),
          "direct_irrep_coefficients": ({"coeffs":np.array([(W[0,0]+W[1,1])/2,(W[1,0]-W[0,1])/2,(W[0,0]-W[1,1])/2,(W[0,1]+W[1,0])/2],np.float32),"parity":np.array([0,1,0,1],np.uint8),"private":targets[4]}, [0,1,2,3,4], {"kind":"direct_irrep_coefficients","views":4,"private_fifth":True}, 4*4+4),
          "direct_coefficients": ({"view_0":targets[0],"view_1":targets[1],"view_2":targets[2],"view_3":targets[3],"private":targets[4]}, [0,1,2,3,4], {"kind":"direct_per_view","views":4,"private_fifth":True}, 5*4*4),
        }
        for name,(arrays,ids,meta,mac) in cases.items():
            started=time.perf_counter()
            payload,loaded,md=pack_payload(arrays,meta)
            if name=="independent": preds=[x@loaded[f"expert_{i}"].T for i in range(5)]
            elif name in ("hard_tied","equivariant_projection"): preds=[x@loaded["expert"].T for _ in range(5)]
            elif name in ("group_mirror","group_mirror_no_private"):
                # The C4 address is paid in payload; decode from metadata's deterministic address order.
                codes=loaded["codes"].astype(int)
                preds=[x@(R[k]@loaded["base"]@R[k].T).T for k in codes]
                preds += [x@loaded["private"].T] if md["private_fifth"] else [x@loaded["base"].T]
            elif name=="direct_irrep_coefficients":
                a,b,c,d=loaded["coeffs"]
                preds=[]
                for parity in loaded["parity"].astype(int):
                    s=1.0 if parity==0 else -1.0
                    matrix=np.array([[a+s*c,-b+s*d],[b+s*d,a-s*c]],dtype=np.float32)
                    preds.append(x@matrix.T)
                preds.append(x@loaded["private"].T)
            else: preds=[x@loaded[f"view_{i}"].T for i in range(4)]+[x@loaded["private"].T]
            errors=[float(np.mean((p-(x@t.T))**2)/(np.mean((x@t.T)**2)+1e-30)) for p,t in zip(preds,targets)]
            # Function identity count at a fixed random probe set and tolerance 1e-5.
            view_outputs=[p[:,:] for p in preds[:4]]
            unique=[]
            for p in view_outputs:
                if not any(np.max(np.abs(p-q))<1e-5 for q in unique): unique.append(p)
            wall=time.perf_counter()-started
            eq_error=0.0
            if name in ("group_mirror","group_mirror_no_private"):
                views=[R[k]@loaded["base"]@R[k].T for k in range(4)]
                eq_error=max(float(np.max(np.abs(R[g]@views[k]@R[g].T-views[(k+g)%4]))) for k in range(4) for g in range(4))
            else:
                eq_error=max(float(np.max(np.abs(R[g]@Weq@R[g].T-Weq))) for g in range(4)) if name=="equivariant_projection" else float("nan")
            # synthetic representation is analytic: zero examples optimized; report required eval compute
            rows.append({"seed":seed,"method":name,"payload_bytes":len(payload),"sha256":hashlib.sha256(payload).hexdigest(),"mean_related_nMSE":float(np.mean(errors[:4])),"max_related_nMSE":max(errors[:4]),"unrelated_nMSE":errors[4],"unique_related_functions":len(unique),"addresses":4,"equivariance_error":eq_error,"active_MACs_per_5_task_inference":int(N*2*2*2*5),"extra_transform_MACs_per_related_view":16 if name.startswith("group_mirror") else 0,"examples_evaluated":N,"optimizer_updates":0,"wall_seconds_serialization_and_eval":wall})
    out=OUT/("development_validation.csv" if args.dev_only else "results.csv")
    with out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(out)
    for r in rows: print(json.dumps(r,sort_keys=True))

if __name__=="__main__": main()
