#!/usr/bin/env python3
"""Gate 7: untouched final-audit eval of frozen candidates (ONE pass)."""
import sys, json, subprocess
from pathlib import Path

ART = {"28M_act_b8_g64": "runs/gate2-controls/28M_act_b8_g64.fqc",
       "28M_act_b4_g64": "runs/gate2-controls/28M_act_b4_g64.fqc",
       "28M_act_b4_g128": "runs/gate2-controls/28M_act_b4_g128.fqc",
       "28M_act_b3_g64": "runs/gate2-controls/28M_act_b3_g64.fqc",
       "28M_act_b3_g128": "runs/gate2-controls/28M_act_b3_g128.fqc",
       "28M_act_b2_g64": "runs/gate2-controls/28M_act_b2_g64.fqc",
       "28M_actmeta_b4_g64_sg128": "runs/gate35-joint/28M_actmeta_b4_g64_sg128.fqc",
       "28M_joint_emb2_attn4_mlp3": "runs/gate35-joint/28M_joint_emb2_attn4_mlp3.fqc",
       "28M_shareK256_B8": "runs/gate3-sharing/28M_shareK256_B8.fqc",
       "28M_vqK256_B8": "runs/gate2-controls/28M_vqK256_B8.fqc",
       "28M_BC_P256": "runs/gate4-private/28M_BC_P256.fqc"}
OUT = Path("runs/eval-001")
rows = []
order = ["ORIGINAL"] + list(ART)
for cand in order:
    out = OUT / f"audit_{cand}.json"
    if out.exists():
        print("AUDIT", cand, "exists - loading prior single-pass result", flush=True)
        e = json.loads(out.read_text())
        s = e["summary"]
        rows.append({"candidate": cand,
                     "NLL_tw": round(s["NLL"]["token_weighted_mean"], 6),
                     "KL_tw": round(s["KL"]["token_weighted_mean"], 6),
                     "agree_tw": round(s["top1_agree"]["token_weighted_mean"], 6),
                     "dNLL_CI": [round(x, 4) for x in s["delta_NLL"]["story_bootstrap_95CI"]]})
        continue
    cmd = [sys.executable, "experiments/gate1/eval_official.py", "--split", "audit",
           "--device", "cpu", "--threads", "4", "--output", str(out)]
    if cand != "ORIGINAL":
        cmd[2:2] = ["--artifact", ART[cand]]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("AUDIT", cand, "rc=", r.returncode, flush=True)
    assert r.returncode == 0, r.stderr[-2000:]
    e = json.loads(out.read_text())
    s = e["summary"]
    rows.append({"candidate": cand,
                 "NLL_tw": round(s["NLL"]["token_weighted_mean"], 6),
                 "KL_tw": round(s["KL"]["token_weighted_mean"], 6),
                 "agree_tw": round(s["top1_agree"]["token_weighted_mean"], 6),
                 "dNLL_CI": [round(x, 4) for x in s["delta_NLL"]["story_bootstrap_95CI"]]})
hdr = "candidate,NLL_tw,KL_tw,agree_tw,dNLL_CI"
(OUT / "GATE7_AUDIT.csv").write_text(hdr + "\n" + "\n".join(
    f"{r['candidate']},{r['NLL_tw']},{r['KL_tw']},{r['agree_tw']},{r['dNLL_CI']}" for r in rows) + "\n")
print(hdr + "\n" + "\n".join(
    f"{r['candidate']},{r['NLL_tw']},{r['KL_tw']},{r['agree_tw']},{r['dNLL_CI']}" for r in rows))
