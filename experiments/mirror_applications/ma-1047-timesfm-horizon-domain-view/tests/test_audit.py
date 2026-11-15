import json
import struct
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"source"))

def test_split_origins_stay_in_declared_regions():
    from run_experiment import make_origins
    manifest=json.loads((ROOT/"runs/INPUT_MANIFEST.json").read_text())
    audit=json.loads((ROOT/"runs/AUDIT_FORECASTS.json").read_text())
    for d,name in enumerate(("ETTh1","ETTm1")):
        n=manifest["datasets"][name]["rows"]
        for split,lo,hi in [
            ("train",512,int(.60*n)-720),
            ("validation",int(.60*n),int(.80*n)-720),
            ("audit",int(.80*n),n-720),
        ]:
            os=make_origins(n,split)
            assert min(os)>=lo and max(os)<=hi
            assert all(o+720<=n for o in os)
        expected=make_origins(n,"audit")
        for ch in range(7):
            assert audit["windows"][f"{d}:{ch}"]["origins"]==expected

def test_saved_audit_metrics_replay_summary():
    rows=json.loads((ROOT/"runs/AUDIT_METRICS.json").read_text())
    summary=json.loads((ROOT/"runs/AUDIT_SUMMARY.json").read_text())
    for ds in ("ETTh1","ETTm1"):
        for method in ("timesfm","shared","additive","independent","mirror_factorized","seasonal_naive"):
            r=[x for x in rows if x["dataset"]==ds and x["method"]==method]
            assert len(r)==28
            assert abs(np.mean([x["mase"] for x in r])-summary[ds][method]["macro_mase"])<1e-12

def test_payload_float32_roundtrip_and_billing():
    manifest=json.loads((ROOT/"runs/INPUT_MANIFEST.json").read_text())
    assert manifest["model"]["common_payload_bytes"]==925181579
    expected={"timesfm":0,"shared":8,"additive":40,"independent":64,"mirror_factorized":48}
    for method,n in expected.items():
        p=ROOT/"payload"/method/"coefficients.f32le"
        data=p.read_bytes()
        assert len(data)==n
        assert len(data)%4==0
        vals=struct.unpack("<"+"f"*(len(data)//4),data) if data else ()
        assert all(np.isfinite(vals))
        assert (struct.pack("<"+"f"*len(vals),*vals) if vals else b"")==data

def test_factorized_view_misses_quality_and_native_control():
    s=json.loads((ROOT/"runs/AUDIT_SUMMARY.json").read_text())
    for ds in ("ETTh1","ETTm1"):
        assert s[ds]["mirror_factorized"]["macro_mase"] >= s[ds]["timesfm"]["macro_mase"]*0.99
        assert abs(s[ds]["mirror_factorized"]["macro_mase"]-s[ds]["additive"]["macro_mase"])<0.01
