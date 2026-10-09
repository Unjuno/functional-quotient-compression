import csv
import hashlib
import importlib.util
import io
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ma476_run", ROOT / "source" / "run.py")
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def test_payloads_and_memory_metrics_replay():
    rows = list(csv.DictReader((ROOT / "RESULTS_CORE.csv").open()))
    assert len(rows) == 288
    assert len({(r["world"], r["seed"]) for r in rows}) == 9
    max_diff = 0.0
    for r in rows:
        w,s,m,n=int(r["world"]),int(r["seed"]),r["method"],int(r["n"])
        keys,radii,basis,ang,values=RUN.make_world(w,s)
        qe,qp,qn=RUN.make_queries(w,s,keys)
        blob=RUN.payload(m,keys,radii,basis,ang,values,n)
        assert len(blob)==int(r["payload_bytes"])
        assert hashlib.sha256(blob).hexdigest()==r["hash"]
        if blob:
            path=ROOT/"artifacts"/"payloads"/f'{w}_{s}_{m}_N{n}.pt'
            assert path.read_bytes()==blob
            obj=torch.load(io.BytesIO(blob),map_location="cpu",weights_only=False)
            b=io.BytesIO();torch.save(obj,b)
            assert b.getvalue()==blob
        o=RUN.load_memory(m,blob)
        out=RUN.evaluate(m,o,qe[:n],qp[:n*4],qn,values[:n])
        cols=("exact_hit_recall","paraphrase_hit_recall","unrelated_false_trigger_rate","exact_value_nrmse","paraphrase_value_nrmse","unrelated_output_drift")
        for a,c in zip(out,cols):
            d=abs(a-float(r[c]));max_diff=max(max_diff,d);assert d<1e-12
    assert max_diff<1e-12


def test_n64_value_quality_storage_frontier():
    rows=list(csv.DictReader((ROOT/"RESULTS_CORE.csv").open()))
    for w in (47610,47611,47612):
        for s in (0,1,2):
            g=[r for r in rows if int(r["world"])==w and int(r["seed"])==s and int(r["n"])==64]
            by={r["method"]:r for r in g}
            for m in ("grace_explicit","generic_coeff","mirror","vq8","vq16","vq32","vq64"):
                assert float(by[m]["exact_hit_recall"])>=.99
                assert float(by[m]["paraphrase_hit_recall"])>=.99
                assert float(by[m]["unrelated_false_trigger_rate"])<=.01
            assert int(by["mirror"]["payload_bytes"])<=.8*int(by["grace_explicit"]["payload_bytes"])
            assert float(by["mirror"]["paraphrase_value_nrmse"])<=1e-5
            for m in ("generic_coeff","vq8","vq16","vq32","vq64"):
                assert int(by["mirror"]["payload_bytes"])<int(by[m]["payload_bytes"])
                assert float(by["mirror"]["paraphrase_value_nrmse"])<float(by[m]["paraphrase_value_nrmse"])+1e-6
