#!/usr/bin/env python3
"""Gate 3: family B = control + FQC functional sharing (per-role codebooks).

Design: same (K, block) as matched family-A per-tensor VQ points, but ONE
codebook is trained and paid per ROLE (embedding / attn_q,k,v,out / mlp_fc /
mlp_proj / position / other) instead of one per tensor. No private exceptions.
Decoder unchanged. Bytes strictly fewer than A by construction; quality is
compared on dev (Pareto vs A frontier and vs uniform frontier).

Fitting: unsupervised kmeans seed 266 (same as A). No eval data touched.
"""
import sys, json, argparse, gc
from pathlib import Path
import torch
sys.path.insert(0, "experiments/t282/code")
from engine import load_checkpoint
from binary_codec import fp16, write_model, train_vq_group, role

ROOT = Path(".")
MODEL_DIR = ROOT / "models/HF-28M"
PAID = 51987968
BASE16 = PAID * 2
TAG = "28M"
CKPT_SHA = "8ddd260f51b439744c8cc785b5516327d4bf32e31ccbfa9009bfadf12557fcf5"
POINTS = [(256, 8), (256, 16), (256, 32), (64, 32)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--paid-scalars", type=int, default=PAID)
    ap.add_argument("--checkpoint-sha", default=CKPT_SHA)
    a = ap.parse_args()
    base16 = a.paid_scalars * 2
    outdir = Path(a.output)
    if outdir.exists() and not a.append:
        raise FileExistsError("refuse to overwrite existing evidence")
    outdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(266)
    cfg, state, _ = load_checkpoint(Path(a.model_dir), "cpu")
    groups = {}
    for key, tensor in state.items():
        if tensor.numpy().ndim == 1:
            continue
        groups.setdefault(role(key), []).append((key, tensor))
    print("role groups:", {r: len(v) for r, v in groups.items()}, flush=True)
    jobs = [(f"b-{K}-{B}", K, B) for K, B in POINTS]
    if a.only:
        jobs = [j for j in jobs if j[0] == a.only]
        assert jobs, "unknown --only"
    rows = []
    for tag, K, B in jobs:
        name = f"{a.tag}_share{'N' if a.normalize else ''}K{K}_B{B}"
        print("BUILD", name, flush=True)
        sections = []
        for key, tensor in state.items():
            if tensor.numpy().ndim == 1:
                desc, blob = {"kind": "fp16", "shape": list(tensor.shape)}, fp16(tensor.numpy()).tobytes()
                sections.append(({**desc, "name": key}, blob))
        for r, items in sorted(groups.items()):
            meta, cbblob, payloads, _ = train_vq_group(
                items, K, block=B, seed=266, normalize_rows=a.normalize)
            cbname = f"cb-shared:{r}"
            sections.append(({"kind": "codebook", "name": cbname, "K": K, "block": B}, cbblob))
            for d, blob in payloads:
                d["codebook"] = cbname
                sections.append((d, blob))
        info = write_model(outdir / (name + ".fqc"), cfg, sections,
                           {"experiment": "GATE3", "family": "B-sharing-role-codebook",
                            "K": K, "block": B, "normalize_rows": a.normalize,
                            "fit": "unsupervised-kmeans-seed266",
                            "checkpoint_sha256": a.checkpoint_sha})
        del sections
        gc.collect()
        rows.append({"candidate": name, **info,
                     "bits_per_paid_scalar": info["bytes"] * 8 / a.paid_scalars,
                     "ratio_vs_16bit": base16 / info["bytes"]})
    man_path = outdir / "GATE3_SHARING.json"
    if a.append and man_path.exists():
        prior = {r["candidate"]: r for r in json.loads(man_path.read_text())}
        prior.update({r["candidate"]: r for r in rows})
        rows = list(prior.values())
    man_path.write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(f"{r['candidate']}: bytes={r['bytes']} ratio={r['ratio_vs_16bit']:.3f}x")


if __name__ == "__main__":
    main()
