import csv
import hashlib
import importlib.util
import io
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("ma481_run",ROOT/"source"/"run.py")
RUN=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(RUN)


def test_payload_hashes_and_retrieval_metrics_replay():
    rows=list(csv.DictReader((ROOT/"RESULTS_CORE.csv").open()))
    assert len(rows)==324
    assert len({(r["world"],r["seed"]) for r in rows})==9
    maxdiff=0.0
    for r in rows:
        w,s,m,n=int(r["world"]),int(r["seed"]),r["method"],int(r["n"])
        basis,angle,keys,values,radii=RUN.make_world(w,s)
        qe,qp,qn=RUN.make_queries(w,s,keys)
        blob=RUN.payload(m,basis,angle,keys,values,radii,n)
        assert len(blob)==int(r["payload_bytes"])
        assert hashlib.sha256(blob).hexdigest()==r["hash"]
        if blob:
            path=ROOT/"artifacts"/"payloads"/f'{w}_{s}_{m}_N{n}.pt'
            assert path.read_bytes()==blob
            obj=torch.load(io.BytesIO(blob),map_location="cpu",weights_only=False)
            buf=io.BytesIO();torch.save(obj,buf);assert buf.getvalue()==blob
        o=RUN.load_mem(blob);decoded=RUN.decode_keys(m,o)
        out=RUN.metrics(m,o,decoded,qe,qp,qn,values,n)
        cols=("exact_correct_address_recall","paraphrase_correct_address_recall","unrelated_false_trigger_rate","exact_output_nrmse","paraphrase_output_nrmse","unrelated_output_drift")
        for a,c in zip(out,cols):
            d=abs(a-float(r[c]));maxdiff=max(maxdiff,d);assert d<1e-12
    assert maxdiff<1e-12


def test_registered_gate_miss_and_vq_collision_boundary():
    rows=list(csv.DictReader((ROOT/"RESULTS_CORE.csv").open()))
    recalls=[]
    for w in (48110,48111,48112):
        for s in (0,1,2):
            g=[r for r in rows if int(r["world"])==w and int(r["seed"])==s and int(r["n"])==64]
            by={r["method"]:r for r in g}
            assert float(by["mirror"]["exact_correct_address_recall"])==1.0
            recalls.append(float(by["mirror"]["paraphrase_correct_address_recall"]))
            assert float(by["explicit"]["paraphrase_correct_address_recall"])==float(by["mirror"]["paraphrase_correct_address_recall"])
            assert int(by["mirror"]["payload_bytes"])<=.7*int(by["explicit"]["payload_bytes"])
            assert int(by["mirror"]["payload_bytes"])<int(by["generic_coeff"]["payload_bytes"])
            assert float(by["vq64"]["address_collision_fraction"])>.4
            assert int(by["vq128"]["payload_bytes"])>int(by["explicit"]["payload_bytes"])
    assert sum(recalls)/len(recalls)<.99
