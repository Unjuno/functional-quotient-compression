#!/usr/bin/env python3
"""Gate 4: family C = sharing + SELECTED private row exceptions (family C).

Backbones: B shareK256_B8 and matched A vqK256_B8 (interaction control).
Support sizes S in {0, 64, 256, 1024} rows, fp16 exact storage, funded openly
(bytes grow; Pareto comparison - nearby-rate rule). S=0 rebuilds the backbone
with C provenance (header-effect control).

Row selection: global top-S calibration-weighted SSE rows (original vs
backbone-decoded), computed from official calibration moments + weight space
only. NO dev/audit eval data used for selection.

New 'rowpatch' section kind; both decoders extended with rejection tests.
"""
import sys, json, argparse, gc
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, "experiments/t282/code")
sys.path.insert(0, "experiments/gate1")
from engine import load_checkpoint, BPETokenizer
from binary_codec import read_model, write_model, encode_rowpatch
from build_control import collect_moments, load_calibration

ROOT = Path(".")
MODEL_DIR = ROOT / "models/HF-28M"
PAID = 51987968
BASE16 = PAID * 2
SUPPORTS = [0, 64, 256, 1024]
BACKBONES = {"B": "runs/gate3-sharing/28M_shareK256_B8.fqc",
             "A": "runs/gate2-controls/28M_vqK256_B8.fqc"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    outdir = Path(a.output)
    if outdir.exists():
        raise FileExistsError("refuse to overwrite existing evidence")
    outdir.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(40604)
    cfg, orig, _ = load_checkpoint(MODEL_DIR, "cpu")
    tok = BPETokenizer(MODEL_DIR)
    moments = collect_moments(orig, cfg, tok, load_calibration())
    rows = []
    for tag, art in BACKBONES.items():
        _, dec, _ = read_model(Path(art), "cpu")
        scored = []
        for key, t in orig.items():
            if t.numpy().ndim != 2:
                continue
            o = t.numpy().astype(np.float64)
            d = dec[key].numpy().astype(np.float64)
            imp = moments.get(key, np.ones(o.shape[1], np.float32)).astype(np.float64)
            sse = ((o - d) ** 2 * imp[None, :]).mean(1)
            for r, s in enumerate(sse):
                scored.append((float(s), key, r))
        scored.sort(reverse=True)
        for S in SUPPORTS:
            name = f"28M_{tag}C_P{S}"
            print("BUILD", name, flush=True)
            by_tensor = {}
            for _, key, r in scored[:S]:
                by_tensor.setdefault(key, []).append(r)
            # verbatim backbone carry: re-read raw sections from artifact file
            import struct as _st
            blob = Path(art).read_bytes()
            _, nh, nb, _ = _st.unpack_from("<8sIQQ", blob)
            meta = json.loads(blob[28:28 + nh])
            body = blob[28 + nh:28 + nh + nb]
            sections = [({"kind": d["kind"], **{k: v for k, v in d.items()
                          if k not in ("offset", "length")}},
                         body[d["offset"]:d["offset"] + d["length"]])
                        for d in meta["sections"]]
            for key, rws in sorted(by_tensor.items()):
                rws = sorted(rws)
                d, b = encode_rowpatch(orig[key].numpy(), rws)
                sections.append(({**d, "name": key}, b))
            info = write_model(outdir / (name + ".fqc"), cfg, sections,
                               {"experiment": "GATE4", "family": "C-private-rowpatch",
                                "backbone": art, "support_rows": S,
                                "fit": "official-calibration-weighted-SSE",
                                "checkpoint_sha256": "8ddd260f51b439744c8cc785b5516327d4bf32e31ccbfa9009bfadf12557fcf5"})
            del sections
            gc.collect()
            rows.append({"candidate": name, **info,
                         "bits_per_paid_scalar": info["bytes"] * 8 / PAID,
                         "ratio_vs_16bit": BASE16 / info["bytes"]})
    (outdir / "GATE4_PRIVATE.json").write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(f"{r['candidate']}: bytes={r['bytes']} ratio={r['ratio_vs_16bit']:.3f}x")


if __name__ == "__main__":
    main()
