import json
import struct
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def test_protocol_freeze_and_pair_holdout():
    p=json.loads((ROOT/"PROTOCOL.json").read_text())
    freeze=json.loads((ROOT/"runs/DEV_FREEZE.json").read_text())
    assert freeze["audit_targets_or_samples_accessed_before_freeze"] is False
    fit={0:{0,2,4,6,8,10,12,13},1:{1,3,5,7,9,11,12,13}}
    audit={0:{1,3,5,7,9,11},1:{0,2,4,6,8,10}}
    assert all(not (fit[f]&audit[f]) for f in [0,1])
    assert all(len(audit[f])==6 for f in [0,1])

def test_fresh_origins_are_inside_final_chronological_split():
    manifest=json.loads((ROOT/"runs/INPUT_MANIFEST.json").read_text())
    for f,h in [(0,288),(1,96)]:
        z=np.load(ROOT/"runs"/f"audit_frequency_{f}.npz")
        n=manifest["data"]["five_minute_rows"] if f==0 else manifest["data"]["fifteen_minute_rows"]
        assert len(z["origins"])==18
        assert np.all(z["origins"]>=int(.8*n))
        assert np.all(z["origins"]+h<=n)
        expected={1,3,5,7,9,11} if f==0 else {0,2,4,6,8,10}
        assert set(z["sensor_ids"].tolist())==expected

def test_metric_summary_replays_and_candidate_fails():
    rows=json.loads((ROOT/"runs/AUDIT_METRICS.json").read_text())
    s=json.loads((ROOT/"runs/AUDIT_SUMMARY.json").read_text())
    assert len(rows)==84
    for f,g in [(0,"5min"),(1,"15min")]:
        for m in ("zero_shot","global","frequency_only","variate_only","additive","mirror_factorized","seasonal_naive"):
            z=[r for r in rows if r["frequency"]==f and r["method"]==m]
            assert len(z)==6
            assert abs(np.mean([r["crps"] for r in z])-s[g][m]["crps"])<1e-12
        assert s[g]["mirror_factorized"]["crps"]>s[g]["zero_shot"]["crps"]
        assert abs(s[g]["mirror_factorized"]["crps"]-s[g]["additive"]["crps"])/s[g]["additive"]["crps"]<0.001

def test_actual_payload_bytes_roundtrip():
    s=json.loads((ROOT/"runs/AUDIT_SUMMARY.json").read_text())
    assert s["mirror_factorized_payload"]["complete_bytes"]==55321371
    assert s["additive_payload"]["complete_bytes"]==55321363
    assert s["independent_full_table_reference_payload"]["quality_result"].startswith("not fit")
    for method,n in {"mirror_factorized":32,"additive":30,"global":2,"frequency_only":4,"variate_only":28}.items():
        data=(ROOT/"payload"/method/"coefficients.f32le").read_bytes()
        assert len(data)==4*n
        vals=struct.unpack("<"+"f"*n,data)
        assert np.isfinite(vals).all()
        assert struct.pack("<"+"f"*n,*vals)==data
