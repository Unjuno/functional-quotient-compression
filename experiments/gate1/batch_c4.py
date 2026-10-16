#!/usr/bin/env python3
"""C4 batch eval: dev or audit over a controls manifest for a given scale.

Usage:
  python experiments/gate1/batch_c4.py --manifest runs/c4-8m-controls/GATE2_CONTROLS.json \
      --artdir runs/c4-8m-controls --model-dir models/HF-8M --split development \
      --outdir runs/c4-8m-eval --tag dev
Appends extra artifacts via --extra dir:prefix (repeatable).
Writes <tag>_FRONTIER.csv. Refuses to overwrite existing outputs.
"""
import sys, json, subprocess, argparse
from pathlib import Path


def run_eval(py, model_dir, artifact, split, threads, out):
    if out.exists():
        return json.loads(out.read_text()), False
    cmd = [py, "experiments/gate1/eval_official.py", "--model-dir", model_dir,
           "--split", split, "--device", "cpu", "--threads", str(threads),
           "--output", str(out)]
    if artifact:
        cmd[2:2] = ["--artifact", artifact]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print("FAIL", out, r.stderr[-1500:], flush=True)
        raise SystemExit(f"eval failed for {out}")
    print("EVAL", out.name, flush=True)
    return json.loads(out.read_text()), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--artdir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--split", required=True, choices=["development", "audit"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--extra", action="append", default=[],
                    help="manifest.json path merged into candidate list")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only-cand", action="append", default=None,
                    help="restrict to these candidate names (repeatable)")
    a = ap.parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cands = [("ORIGINAL", None)]
    man = json.loads(Path(a.manifest).read_text())
    artdir = Path(a.artdir)
    for m in man:
        cands.append((m["candidate"], str(artdir / (m["candidate"] + ".fqc"))))
    for ex in a.extra:
        mp, prefix = ex.split(":")
        for m in json.loads(Path(mp).read_text()):
            cands.append((m["candidate"], None))  # resolved below
    # resolve extra artifact paths by candidate name search
    resolved = []
    for cand, art in cands:
        if art is None and cand != "ORIGINAL":
            hits = sorted(Path(".").glob(f"runs/*/{cand}.fqc"))
            assert len(hits) == 1, f"artifact lookup failed for {cand}: {hits}"
            art = str(hits[0])
        resolved.append((cand, art))
    rows = []
    for cand, art in resolved:
        if a.only_cand and cand not in a.only_cand:
            continue
        out = outdir / f"{a.tag}_{cand}.json"
        e, _ = run_eval(sys.executable, a.model_dir, art, a.split, a.threads, out)
        s = e["summary"]
        rows.append({"candidate": cand,
                     "NLL_tw": round(s["NLL"]["token_weighted_mean"], 6),
                     "PPL": round(s["perplexity"], 4),
                     "KL_tw": round(s["KL"]["token_weighted_mean"], 6),
                     "dNLL_tw": round(s["delta_NLL"]["token_weighted_mean"], 6),
                     "agree_tw": round(s["top1_agree"]["token_weighted_mean"], 6),
                     "dNLL_CI": [round(x, 4) for x in s["delta_NLL"]["story_bootstrap_95CI"]]})
    csv = outdir / f"{a.tag}_FRONTIER.csv"
    if csv.exists():
        raise FileExistsError("refuse to overwrite frontier csv")
    lines = ["candidate,NLL_tw,PPL,KL_tw,dNLL_tw,agree_tw,dNLL_CI"] + [
        f"{r['candidate']},{r['NLL_tw']},{r['PPL']},{r['KL_tw']},{r['dNLL_tw']},{r['agree_tw']},{r['dNLL_CI']}"
        for r in rows]
    csv.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
