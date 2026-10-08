#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"artifacts"
SEEDS=(33701,33702,33711,33712,33713)
N=2048
SEEN=[(p,r) for p in range(4) for r in range(2) if (p,r) not in ((2,1),(3,1))]
HELD=[(2,1),(3,1)]
ALL_PAIRS=[(p,r) for p in range(4) for r in range(2)]

def actions():
    P=np.roll(np.eye(4,dtype=np.float32),1,axis=0)
    S=np.array([[0,1],[1,0]],np.float32)
    return [np.linalg.matrix_power(P,p) for p in range(4)],[np.linalg.matrix_power(S,r) for r in range(2)]

def encode(arrays,meta):
    bio=io.BytesIO()
    with zipfile.ZipFile(bio,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in sorted(arrays):
            b=io.BytesIO();np.save(b,np.asarray(arrays[name]),allow_pickle=False)
            i=zipfile.ZipInfo(name+".npy",(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o600<<16
            z.writestr(i,b.getvalue())
        i=zipfile.ZipInfo("metadata.json",(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o600<<16
        z.writestr(i,json.dumps(meta,sort_keys=True,separators=(",",":")).encode())
    payload=bio.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        loaded={k[:-4]:np.load(io.BytesIO(z.read(k)),allow_pickle=False) for k in z.namelist() if k.endswith(".npy")}
        md=json.loads(z.read("metadata.json"))
    return payload,loaded,md

def world(seed):
    rng=np.random.default_rng(seed); W=rng.normal(size=(8,8)).astype(np.float32)
    Ps,Ss=actions()
    target={(p,r):np.kron(Ps[p],Ss[r])@W@np.kron(Ps[p],Ss[r]).T for p in range(4) for r in range(2)}
    off=rng.normal(size=(8,8)).astype(np.float32)+np.diag(np.arange(1,9,dtype=np.float32))
    x=rng.normal(size=(N,8)).astype(np.float32)
    return W,target,off,x,Ps,Ss

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--dev-only",action="store_true");args=ap.parse_args()
    OUT.mkdir(exist_ok=True); rows=[]
    for seed in SEEDS[:2] if args.dev_only else SEEDS:
        W,targ,off,x,Ps,Ss=world(seed)
        group=np.kron(Ps[0],Ss[0]); _=group
        mats_by=[targ[pair] for pair in ALL_PAIRS]
        eq=np.mean([np.kron(Ps[p],Ss[r])@W@np.kron(Ps[p],Ss[r]).T for p in range(4) for r in range(2)],axis=0)
        cases={
          "independent_8":({f"op{i}":m for i,m in enumerate(mats_by)},{"kind":"independent","count":8}),
          "independent_9_with_offorbit":({**{f"op{i}":m for i,m in enumerate(mats_by)},"private":off},{"kind":"independent","count":9}),
          "flat_seen_6":({f"op{i}":targ[p,r] for i,(p,r) in enumerate(SEEN)},{"kind":"flat_table","seen":6}),
          "hard_tied":({"op":W},{"kind":"hard_tied"}),
          "equivariant_projection":({"op":eq},{"kind":"equivariant"}),
          "factorized_mirror":({"base":W,"pos":np.array([p for p,r in ALL_PAIRS],np.uint8),"role":np.array([r for p,r in ALL_PAIRS],np.uint8)},{"kind":"group_action","group":"C4xC2","pairs":ALL_PAIRS,"private":False}),
          "direct_coordinates":({"base":W,"pos":np.array([p for p,r in ALL_PAIRS],np.uint8),"role":np.array([r for p,r in ALL_PAIRS],np.uint8)},{"kind":"native_direct_coordinates","group":"C4xC2","pairs":ALL_PAIRS,"private":False}),
          "factorized_with_private":({"base":W,"pos":np.array([p for p,r in ALL_PAIRS],np.uint8),"role":np.array([r for p,r in ALL_PAIRS],np.uint8),"private":off},{"kind":"group_action","group":"C4xC2","pairs":ALL_PAIRS,"private":True}),
          "factorized_no_private_offorbit":({"base":W,"pos":np.array([p for p,r in ALL_PAIRS],np.uint8),"role":np.array([r for p,r in ALL_PAIRS],np.uint8)},{"kind":"group_action","group":"C4xC2","pairs":ALL_PAIRS,"private":False})}
        for name,(arrays,meta) in cases.items():
            start=time.perf_counter();payload,ld,md=encode(arrays,meta)
            if name in ("independent_8","independent_9_with_offorbit"):
                preds=[x@ld[f"op{i}"].T for i in range(8)]
                if name=="independent_9_with_offorbit": preds.append(x@ld["private"].T)
            elif name=="flat_seen_6": preds=[x@ld[f"op{i}"].T for i in range(6)]
            elif name in ("hard_tied","equivariant_projection"): preds=[x@ld["op"].T for _ in range(8)]
            else:
                preds=[]
                for p,r in zip(ld["pos"].astype(int),ld["role"].astype(int)):
                    A=np.kron(Ps[p],Ss[r]);preds.append(x@(A@ld["base"]@A.T).T)
                if "private" in ld: preds.append(x@ld["private"].T)
                else: preds.append(x@ld["base"].T)
            elapsed=time.perf_counter()-start
            # Evaluation quality for the eight compositions; table methods lacking held-out rows are marked unavailable.
            if len(preds)>=8: errs=[float(np.mean((q-x@targ[p,r].T)**2)/(np.mean((x@targ[p,r].T)**2)+1e-30)) for q,(p,r) in zip(preds[:8],[(p,r) for p in range(4) for r in range(2)])]
            else: errs=[float("nan")]*8
            distinct=[]
            for q in preds[:8]:
                if not any(np.max(np.abs(q-q0))<1e-5 for q0 in distinct):distinct.append(q)
            held=[errs[5],errs[7]] if len(preds)>=8 else [float("nan")]*2
            offerr=float(np.mean((preds[8]-x@off.T)**2)/(np.mean((x@off.T)**2)+1e-30)) if len(preds)>8 else float("nan")
            finite=[e for e in errs if np.isfinite(e)];meanerr=float(np.mean(finite)) if finite else float("nan")
            rows.append({"seed":seed,"method":name,"payload_bytes":len(payload),"sha256":hashlib.sha256(payload).hexdigest(),"mean_composition_nMSE":meanerr,"heldout_composition_nMSE":json.dumps(held),"offorbit_nMSE":offerr,"unique_compositions":len(distinct),"examples":N,"optimizer_updates":0,"active_MACs_8_compositions":N*8*8*8,"extra_transform_MACs_per_composition":2*8*8 if name.startswith("factorized") or name=="direct_coordinates" else 0,"wall_seconds_serialization_eval":elapsed})
    out=OUT/("development.csv" if args.dev_only else "results.csv")
    with out.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    for r in rows:print(json.dumps(r,sort_keys=True))
if __name__=="__main__":main()
