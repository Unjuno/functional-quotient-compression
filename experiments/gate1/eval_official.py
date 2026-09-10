#!/usr/bin/env python3
"""Gate 1/2 shared harness: official-validation splits + token-weighted NLL / PPL /
KL(orig||comp) / top-1 agreement, story-level bootstrap CIs.

Splits (preregistered, primary corpus TinyStories-valid.txt, 21990 stories):
  calibration : stories [0, 64)     - activation moments / refit fitting only
  development : stories [64, 320)   - candidate selection only (256 stories)
  audit       : stories [320, 1344) - untouched until Gate 7 (1024 stories)

Usage:
  python experiments/gate1/eval_official.py --artifact runs/rebuild-001/28M_act_b4_g128.fqc \
      --split development --device cpu --threads 4 --output runs/eval-001/dev_act_b4_g128.json
  (no --artifact => original checkpoint as reference run)
"""
import sys, json, argparse, time, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, "experiments/t282/code")
from engine import BPETokenizer, forward, load_checkpoint
from binary_codec import read_model

ROOT = Path(".")
PRIMARY = ROOT / "data/official/TinyStories-valid.txt"
CAL = (0, 64)
DEV = (64, 320)
AUDIT = (320, 1344)
BOOT = 4000
SEED = 266272


def load_stories():
    raw = PRIMARY.read_text()
    return [s.strip() for s in raw.split("<|endoftext|>") if s.strip()]


def story_metrics(ids, ref_logits, cmp_logits):
    """ids: [T] long; logits: [T, V] float (already sliced to :-1 outside)."""
    tgt = ids[1:]
    lr = F.log_softmax(ref_logits, dim=-1)
    lc = F.log_softmax(cmp_logits, dim=-1)
    kl = (lr.exp() * (lr - lc)).sum(-1)
    nll = F.nll_loss(lc, tgt, reduction="none")
    ref_nll = F.nll_loss(lr, tgt, reduction="none")
    flip = (ref_logits.argmax(-1) != cmp_logits.argmax(-1)).float()
    return {"n_tokens": int(len(tgt)), "KL": float(kl.mean()),
            "NLL": float(nll.mean()), "reference_NLL": float(ref_nll.mean()),
            "delta_NLL": float((nll - ref_nll).mean()),
            "top1_flip": float(flip.mean()),
            "top1_agree": float(1.0 - flip.mean()),
            "finite": bool(torch.isfinite(cmp_logits).all() and torch.isfinite(kl).all())}


def summarize(rows):
    rng = np.random.default_rng(SEED)
    inds = rng.integers(0, len(rows), size=(BOOT, len(rows)))
    n_tok = sum(r["n_tokens"] for r in rows)
    tw = {}
    for key in ["KL", "NLL", "reference_NLL", "delta_NLL", "top1_flip", "top1_agree"]:
        a = np.array([r[key] for r in rows])
        w = np.array([r["n_tokens"] for r in rows], dtype=np.float64)
        tw_mean = float((a * w).sum() / w.sum())
        means = a[inds].mean(1)
        tw[key] = {"token_weighted_mean": tw_mean,
                   "story_mean": float(a.mean()),
                   "story_bootstrap_95CI": [float(x) for x in np.quantile(means, [0.025, 0.975])],
                   "story_se": float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else None}
    tw["n_stories"] = len(rows)
    tw["n_tokens"] = n_tok
    tw["all_finite"] = all(r["finite"] for r in rows)
    nll = tw["NLL"]["token_weighted_mean"]
    tw["perplexity"] = float(np.exp(nll))
    return {"per_story": rows, "summary": tw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--split", required=True, choices=["calibration", "development", "audit"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-stories", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError("refuse to overwrite existing evidence")
    torch.set_num_threads(a.threads)
    lo, hi = {"calibration": CAL, "development": DEV, "audit": AUDIT}[a.split]
    stories = load_stories()
    assert len(stories) == 21990, f"primary corpus changed: {len(stories)}"
    if a.max_stories:
        hi = min(hi, lo + a.max_stories)
    stories = stories[lo:hi]
    model_dir = ROOT / "models/HF-28M"
    cfg, ref_state, _ = load_checkpoint(model_dir, "cpu")
    tok = BPETokenizer(model_dir)
    if a.artifact:
        _, cmp_state, meta = read_model(Path(a.artifact), "cpu")
        art_sha = hashlib.file_digest(Path(a.artifact).open("rb"), "sha256").hexdigest()
    else:
        cmp_state, meta, art_sha = ref_state, {"candidate": "ORIGINAL_FP32"}, None
    ref_state = {k: v.to(a.device) for k, v in ref_state.items()}
    cmp_state = {k: v.to(a.device) for k, v in cmp_state.items()}
    rows = []
    t = time.perf_counter()
    with torch.inference_mode():
        for i, text in enumerate(stories):
            ids = torch.tensor(tok.encode(text))
            if len(ids) < 2 or len(ids) > cfg["max_position_embeddings"]:
                continue
            inp = ids[None].to(a.device)
            r = forward(ref_state, cfg, inp)[0, :-1]
            c = forward(cmp_state, cfg, inp)[0, :-1]
            row = story_metrics(ids, r.cpu(), c.cpu())
            row["story_index"] = lo + i
            rows.append(row)
    result = {"split": a.split, "range": [lo, hi], "n": len(rows),
              "artifact": a.artifact, "artifact_sha256": art_sha,
              "device": a.device, "threads": a.threads,
              "elapsed_seconds": time.perf_counter() - t,
              **summarize(rows)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    s = result["summary"]
    print(json.dumps({"split": a.split, "n": len(rows),
                      "NLL_tw": s["NLL"]["token_weighted_mean"],
                      "PPL": s["perplexity"],
                      "KL_tw": s["KL"]["token_weighted_mean"],
                      "agree_tw": s["top1_agree"]["token_weighted_mean"],
                      "dNLL_storyCI": s["delta_NLL"]["story_bootstrap_95CI"]}, indent=2))


if __name__ == "__main__":
    main()
