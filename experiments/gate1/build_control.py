#!/usr/bin/env python3
"""Gate 2: strong activation-aware NON-SHARING control frontier (family A).

Uniform affine grid (rtn + act refit) for ~2x-7.5x plus per-tensor VQ
(non-sharing: one codebook paid per tensor, no cross-tensor reuse) for ~8x-100x.

Fitting inputs: official primary calibration stories [0, 64) ONLY.
Selection inputs: development stories [64, 320) (separate step).
Audit stories [320, 1344) are NEVER touched here.

Writes .fqc artifacts + manifest with ACTUAL serialized bytes.
Deterministic: fixed seeds; CPU FP64 moments (same recipe as T277 rebuild).
"""
import sys, json, argparse, hashlib, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, "experiments/t282/code")
from engine import BPETokenizer, forward, load_checkpoint
from binary_codec import encode_uniform, fp16, write_model, train_vq_group
from affine_refit import encode_refit

ROOT = Path(".")
MODEL_DIR = ROOT / "models/HF-28M"
SEED = 273278
PAID_SCALARS = 51987968
BASE16 = PAID_SCALARS * 2
TAG = "28M"
CKPT_SHA = "8ddd260f51b439744c8cc785b5516327d4bf32e31ccbfa9009bfadf12557fcf5"

UNIFORM_GRID = [(8, 64), (6, 64), (4, 64), (4, 128), (3, 64), (3, 128), (2, 64), (2, 256)]
VQ_GRID = [(256, 4), (256, 8), (64, 8), (256, 16), (256, 32), (64, 32)]


def load_calibration():
    raw = (ROOT / "data/official/TinyStories-valid.txt").read_text()
    stories = [s.strip() for s in raw.split("<|endoftext|>") if s.strip()]
    assert len(stories) == 21990, "primary corpus changed"
    return stories[:64]


def collect_moments(state, cfg, tok, stories):
    mapping = {v.data_ptr(): k for k, v in state.items() if v.ndim == 2}
    sums, counts = {}, {}
    orig = F.linear

    def observed(x, weight, bias=None):
        name = mapping.get(weight.data_ptr())
        if name and (".mlp." in name or ".attn." in name):
            flat = x.detach().reshape(-1, x.shape[-1])
            ss = (flat.double() * flat.double()).sum(0).cpu()
            if name not in sums:
                sums[name], counts[name] = ss, len(flat)
            else:
                sums[name] += ss
                counts[name] += len(flat)
        return orig(x, weight, bias)

    F.linear = observed
    try:
        with torch.inference_mode():
            for text in stories:
                forward(state, cfg, torch.tensor([tok.encode(text)], device="cpu"))
    finally:
        F.linear = orig
    out = {}
    for name, total in sums.items():
        a = (total / counts[name]).numpy()
        a = np.maximum(a, max(float(a.mean()) * 1e-8, 1e-12))
        a /= a.mean()
        out[name] = a.astype(np.float32)
    assert len(out) == 6 * cfg["num_layers"], "incomplete moment collection"
    return out


def build_uniform(state, cfg, moments, bits, group, method, outdir,
                  tag=TAG, paid=PAID_SCALARS, ckpt=CKPT_SHA):
    name = f"{tag}_{method}_b{bits}_g{group}"
    sections = []
    for key, tensor in state.items():
        a = tensor.numpy()
        if a.ndim == 1:
            desc, blob = {"kind": "fp16", "shape": list(a.shape)}, fp16(a).tobytes()
        elif method == "rtn":
            desc, blob = encode_uniform(a, bits, group)
        else:
            desc, blob, _ = encode_refit(a, bits, moments.get(key), group=group)
        sections.append(({**desc, "name": key}, blob))
    info = write_model(outdir / (name + ".fqc"), cfg, sections,
                       {"experiment": "GATE2", "family": "A-control",
                        "method": method, "bits": bits, "group": group,
                        "fit": "official-calibration-0-64",
                        "checkpoint_sha256": ckpt})
    del sections
    gc.collect()
    return {"candidate": name, **info,
            "bits_per_paid_scalar": info["bytes"] * 8 / paid,
            "ratio_vs_16bit": (paid * 2) / info["bytes"]}


def build_vq(state, cfg, K, block, outdir, normalize=False,
             tag=TAG, paid=PAID_SCALARS, ckpt=CKPT_SHA):
    vqtag = f"{tag}_vq{'N' if normalize else ''}K{K}_B{block}"
    name = vqtag.replace(" ", "")
    sections, stats = [], []
    for key, tensor in state.items():
        a = tensor.numpy()
        if a.ndim == 1:
            desc, blob = {"kind": "fp16", "shape": list(a.shape)}, fp16(a).tobytes()
            sections.append(({**desc, "name": key}, blob))
            continue
        meta, cbblob, payloads, st = train_vq_group(
            [(key, tensor)], K, block=block, seed=266, normalize_rows=normalize)
        cbname = f"cb:{key}"
        sections.append(({"kind": "codebook", "name": cbname,
                          "K": K, "block": block}, cbblob))
        for d, blob in payloads:
            d["codebook"] = cbname
            sections.append((d, blob))
        stats.extend(st)
    info = write_model(outdir / (name + ".fqc"), cfg, sections,
                       {"experiment": "GATE2", "family": "A-control-VQ-per-tensor",
                        "K": K, "block": block, "normalize_rows": normalize,
                        "fit": "unsupervised-kmeans-seed266",
                        "checkpoint_sha256": ckpt})
    del sections
    gc.collect()
    return {"candidate": name, **info,
            "bits_per_paid_scalar": info["bytes"] * 8 / paid,
            "ratio_vs_16bit": (paid * 2) / info["bytes"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--append", action="store_true",
                    help="allow adding artifacts to an existing output dir")
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--paid-scalars", type=int, default=PAID_SCALARS)
    ap.add_argument("--checkpoint-sha", default=CKPT_SHA)
    ap.add_argument("--skip", action="append", default=[],
                    help="skip job keys (repeatable); structural inapplicability only, recorded in log")
    a = ap.parse_args()
    outdir = Path(a.output)
    if outdir.exists() and not a.append:
        raise FileExistsError("refuse to overwrite existing evidence")
    outdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    model_dir = Path(a.model_dir)
    cfg, state, _ = load_checkpoint(model_dir, "cpu")
    # Header recipe fidelity: store the full config as shipped (including
    # `_name_or_path`), exactly like the T277 rebuild. Verified: with T277
    # provenance labels this reproduces the historical header byte count
    # (16471) to the byte; payload equality already confirmed (29303584).
    # The +56 header bytes vs history come solely from the GATE2 provenance
    # labels below and are paid/counted in every artifact of this lane.
    tok = BPETokenizer(model_dir)
    stories = load_calibration()
    assert all(tok.decode(tok.encode(s)) == s for s in stories[:8]), "tokenizer drift"
    moments = collect_moments(state, cfg, tok, stories)
    rows = []
    cols = sorted({t.shape[1] for t in state.values() if t.numpy().ndim == 2})
    grid = [(b, g) for b, g in UNIFORM_GRID if all(c % g == 0 for c in cols)]
    skipped = [(b, g) for b, g in UNIFORM_GRID if not all(c % g == 0 for c in cols)]
    for b, g in skipped:
        print(f"SKIP u-*-{b}-{g}: group {g} incompatible with 2D cols {cols} "
              f"(refit requires col % group == 0; pair kept paired)", flush=True)
    jobs = [(f"u-{m}-{b}-{g}", m, b, g) for m in ("rtn", "act") for b, g in grid]
    jobs += [(f"vq-{K}-{B}", K, B) for K, B in VQ_GRID]
    jobs += [(f"vqn-{K}-{B}", K, B) for K, B in [(256, 8), (256, 16), (256, 32), (64, 32)]]
    if a.only:
        jobs = [j for j in jobs if j[0] == a.only]
        assert jobs, "unknown --only key"
    for sk in (a.skip or []):
        jobs = [j for j in jobs if j[0] != sk]
        print(f"SKIP {sk}: declared structurally inapplicable for this scale", flush=True)
    for job in jobs:
        print("BUILD", job[0], flush=True)
        if job[0].startswith("u-"):
            _, m, b, g = job
            rows.append(build_uniform(state, cfg, moments, b, g, m, outdir,
                                      tag=a.tag, paid=a.paid_scalars, ckpt=a.checkpoint_sha))
        elif job[0].startswith("vqn-"):
            _, K, B = job
            rows.append(build_vq(state, cfg, K, B, outdir, normalize=True,
                                 tag=a.tag, paid=a.paid_scalars, ckpt=a.checkpoint_sha))
        else:
            _, K, B = job
            rows.append(build_vq(state, cfg, K, B, outdir,
                                 tag=a.tag, paid=a.paid_scalars, ckpt=a.checkpoint_sha))
    man_path = outdir / "GATE2_CONTROLS.json"
    if a.append and man_path.exists():
        prior = {r["candidate"]: r for r in json.loads(man_path.read_text())}
        prior.update({r["candidate"]: r for r in rows})
        rows = list(prior.values())
    man_path.write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(f"{r['candidate']}: bytes={r['bytes']} ratio={r['ratio_vs_16bit']:.3f}x bpp={r['bits_per_paid_scalar']:.3f}")


if __name__ == "__main__":
    main()
