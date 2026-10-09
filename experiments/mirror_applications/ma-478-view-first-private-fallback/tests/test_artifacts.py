import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ma478_run", ROOT / "source" / "run.py")
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def test_fresh_payload_and_metrics_replay():
    rows = list(csv.DictReader((ROOT / "RESULTS_CORE.csv").open()))
    assert len(rows) == 252
    assert json.loads((ROOT / "artifacts" / "development_selection.json").read_text())["tau"] == .3
    max_diff = 0.0
    for r in rows:
        w,s,m,n,tau=int(r["world"]),int(r["seed"]),r["method"],int(r["n"]),float(r["tau"])
        keys,radii,basis,ang,vals,aligned=RUN.make_world(w,s)
        qe,qp,qn=RUN.make_queries(w,s,keys)
        blob=RUN.make_payload(m,keys,radii,basis,vals,aligned,n,tau)
        assert len(blob)==int(r["payload_bytes"])
        assert hashlib.sha256(blob).hexdigest()==r["hash"]
        if blob:
            path=ROOT/"artifacts"/"payloads"/f'{w}_{s}_{m}_N{n}.pt'
            assert path.read_bytes()==blob
            obj=torch.load(io.BytesIO(blob),map_location="cpu",weights_only=False)
            b=io.BytesIO();torch.save(obj,b)
            assert b.getvalue()==blob
        o=RUN.load_memory(blob)
        out=RUN.evaluate(m,o,qe[:n],qp[:n*4],qn,vals[:n])
        cols=("exact_hit_recall","paraphrase_hit_recall","unrelated_false_trigger_rate","exact_value_nrmse","paraphrase_value_nrmse","unrelated_output_drift")
        for a,c in zip(out,cols):
            d=abs(a-float(r[c]));max_diff=max(max_diff,d);assert d<1e-12
    assert max_diff<1e-12


def test_threshold_fallback_boundary_and_failed_byte_gate():
    rows=list(csv.DictReader((ROOT/"RESULTS_CORE.csv").open()))
    for w in (47810,47811,47812):
        for s in (0,1,2):
            g=[r for r in rows if int(r["world"])==w and int(r["seed"])==s and int(r["n"])==64]
            by={r["method"]:r for r in g}
            assert float(by["mirror_fallback"]["fallback_fraction"])==19/64
            assert float(by["mirror_fallback"]["exact_value_nrmse"])<1e-5
            assert float(by["mirror_fallback"]["paraphrase_value_nrmse"])<1e-5
            assert float(by["mirror_only"]["exact_value_nrmse"])>.15
            assert int(by["mirror_fallback"]["payload_bytes"])>.8*int(by["explicit"]["payload_bytes"])
            assert int(by["mirror_fallback"]["payload_bytes"])<int(by["explicit"]["payload_bytes"])
            assert int(by["mirror_fallback"]["payload_bytes"])<int(by["generic_fallback"]["payload_bytes"])
            assert float(by["oracle_fallback"]["exact_value_nrmse"])<1e-5
