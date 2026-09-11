#!/usr/bin/env python3
"""Viable-regime sharing + joint allocation (B-meta and D-lite).

B-meta: uniform b4 codes at group 64 with fp16 (offset, scale) SHARED per
supergroup of 128 row-major elements (act-weighted fit). Tests whether
metadata sharing beats the uniform Pareto line in the viable regime.

D-lite: per-module mixed precision (emb / attn / mlp bit triple, act-refit,
g64) evaluated as JOINT decoded models at nearby bytes to act_b3_g64.
Joint selection happens on dev (separate step); this script only builds.

Fitting: official calibration stories [0,64). No dev/audit data touched.
"""
import sys, json, argparse, gc
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, "experiments/t282/code")
sys.path.insert(0, "experiments/gate1")
from engine import load_checkpoint, BPETokenizer
from binary_codec import fp16, write_model, encode_uniform, encode_sharedmeta
from affine_refit import encode_refit
from build_control import collect_moments, load_calibration

ROOT = Path(".")
MODEL_DIR = ROOT / "models/HF-28M"
PAID = 51987968
TAG = "28M"
CKPT = "8ddd260f51b439744c8cc785b5516327d4bf32e31ccbfa9009bfadf12557fcf5"


def module_of(key):
    if "wte" in key or "wpe" in key:
        return "emb"
    if ".attn." in key:
        return "attn"
    if ".mlp." in key:
        return "mlp"
    return "other"


def build_sharedmeta(state, cfg, moments, outdir, tag=TAG, paid=PAID, ckpt=CKPT):
    name = f"{tag}_actmeta_b4_g64_sg128"
    print("BUILD", name, flush=True)
    fallbacks = []
    sections = []
    for key, tensor in state.items():
        a = tensor.numpy()
        if a.ndim == 1:
            desc, blob = {"kind": "fp16", "shape": list(a.shape)}, fp16(a).tobytes()
        else:
            try:
                imp = moments.get(key)
                w = None if imp is None else np.tile(np.asarray(imp, np.float32), a.shape[0])[:a.size]
                desc, blob = encode_sharedmeta(a, 4, 64, 128, importance=w)
            except ValueError:
                desc, blob, _ = encode_refit(a, 4, moments.get(key), group=64)
                desc = {**desc, "fallback": "uniform-refit"}
                fallbacks.append(key)
        sections.append(({**desc, "name": key}, blob))
    info = write_model(outdir / (name + ".fqc"), cfg, sections,
                       {"experiment": "GATE3B", "family": "B-sharing-metadata",
                        "bits": 4, "group": 64, "supergroup": 128,
                        "fit": "official-calibration-weighted", "checkpoint_sha256": ckpt,
                        "fallback_tensors": fallbacks})
    del sections
    gc.collect()
    return {"candidate": name, **info, "bits_per_paid_scalar": info["bytes"] * 8 / paid,
            "ratio_vs_16bit": (paid * 2) / info["bytes"]}


def build_joint(state, cfg, moments, triple, outdir, tag=TAG, paid=PAID, ckpt=CKPT):
    name = f"{tag}_joint_emb{triple['emb']}_attn{triple['attn']}_mlp{triple['mlp']}"
    print("BUILD", name, flush=True)
    sections = []
    for key, tensor in state.items():
        a = tensor.numpy()
        if a.ndim == 1:
            desc, blob = {"kind": "fp16", "shape": list(a.shape)}, fp16(a).tobytes()
        else:
            b = triple[module_of(key)]
            desc, blob, _ = encode_refit(a, b, moments.get(key), group=64)
        sections.append(({**desc, "name": key}, blob))
    info = write_model(outdir / (name + ".fqc"), cfg, sections,
                       {"experiment": "GATE5-DLITE", "family": "D-joint-mixed-precision",
                        "triple": triple, "group": 64,
                        "fit": "official-calibration-weighted", "checkpoint_sha256": CKPT})
    del sections
    gc.collect()
    return {"candidate": name, **info, "bits_per_paid_scalar": info["bytes"] * 8 / PAID,
            "ratio_vs_16bit": BASE16 / info["bytes"]}


    info = write_model(outdir / (name + ".fqc"), cfg, sections,
                       {"experiment": "GATE5-DLITE", "family": "D-joint-mixed-precision",
                        "triple": triple, "group": 64,
                        "fit": "official-calibration-weighted", "checkpoint_sha256": ckpt})
    del sections
    gc.collect()
    return {"candidate": name, **info, "bits_per_paid_scalar": info["bytes"] * 8 / paid,
            "ratio_vs_16bit": (paid * 2) / info["bytes"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--only", default=None, help="meta or joint0/1/2")
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--paid-scalars", type=int, default=PAID)
    ap.add_argument("--checkpoint-sha", default=CKPT)
    a = ap.parse_args()
    outdir = Path(a.output)
    if outdir.exists():
        raise FileExistsError("refuse to overwrite existing evidence")
    outdir.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(50711)
    model_dir = Path(a.model_dir)
    cfg, state, _ = load_checkpoint(model_dir, "cpu")
    tok = BPETokenizer(model_dir)
    moments = collect_moments(state, cfg, tok, load_calibration())
    triples = ({"emb": 4, "attn": 3, "mlp": 2},
               {"emb": 4, "attn": 2, "mlp": 3},
               {"emb": 2, "attn": 4, "mlp": 3})
    rows = []
    if a.only in (None, "meta"):
        rows.append(build_sharedmeta(state, cfg, moments, outdir,
                                     tag=a.tag, paid=a.paid_scalars, ckpt=a.checkpoint_sha))
    for i, triple in enumerate(triples):
        if a.only in (None, f"joint{i}"):
            rows.append(build_joint(state, cfg, moments, triple, outdir,
                                    tag=a.tag, paid=a.paid_scalars, ckpt=a.checkpoint_sha))
    assert rows, "unknown --only key"
    (outdir / "GATE35_JOINT.json").write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(f"{r['candidate']}: bytes={r['bytes']} ratio={r['ratio_vs_16bit']:.3f}x")


if __name__ == "__main__":
    main()
