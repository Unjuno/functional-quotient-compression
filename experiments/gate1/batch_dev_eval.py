#!/usr/bin/env python3
"""Batch dev-split eval over Gate-2 control artifacts -> frontier CSV."""
import sys, json, subprocess
from pathlib import Path

CTRL = Path("runs/gate2-controls")
OUT = Path("runs/eval-001")
MAN = json.loads((CTRL / "GATE2_CONTROLS.json").read_text())
rows = []
for m in MAN:
    cand = m["candidate"]
    art = str(CTRL / (cand + ".fqc"))
    out = OUT / f"dev_{cand}.json"
    if not out.exists():
        r = subprocess.run([sys.executable, "experiments/gate1/eval_official.py",
                            "--artifact", art, "--split", "development",
                            "--device", "cpu", "--threads", "4",
                            "--output", str(out)],
                           capture_output=True, text=True)
        print("EVAL", cand, "rc=", r.returncode, flush=True)
        if r.returncode:
            print(r.stderr[-2000:])
            continue
    e = json.loads(out.read_text())
    s = e["summary"]
    rows.append({"candidate": cand, "bytes": m["bytes"],
                 "ratio": round(m["ratio_vs_16bit"], 3),
                 "bpp": round(m["bits_per_paid_scalar"], 3),
                 "NLL_tw": round(s["NLL"]["token_weighted_mean"], 6),
                 "PPL": round(s["perplexity"], 4),
                 "KL_tw": round(s["KL"]["token_weighted_mean"], 6),
                 "dNLL_tw": round(s["delta_NLL"]["token_weighted_mean"], 6),
                 "agree_tw": round(s["top1_agree"]["token_weighted_mean"], 6),
                 "dNLL_CI": [round(x, 4) for x in s["delta_NLL"]["story_bootstrap_95CI"]]})
hdr = "candidate,bytes,ratio,bpp,NLL_tw,PPL,KL_tw,dNLL_tw,agree_tw,dNLL_CI"
lines = [hdr] + [f"{r['candidate']},{r['bytes']},{r['ratio']},{r['bpp']},{r['NLL_tw']},{r['PPL']},{r['KL_tw']},{r['dNLL_tw']},{r['agree_tw']},{r['dNLL_CI']}" for r in rows]
(OUT / "GATE2_DEV_FRONTIER.csv").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
