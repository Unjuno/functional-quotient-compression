import csv
import hashlib
import importlib.util
import io
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ma475_run", ROOT / "source" / "run.py")
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def test_payload_and_retrieval_metrics_replay():
    rows = list(csv.DictReader((ROOT / "RESULTS_CORE.csv").open()))
    assert len(rows) == 144
    assert len({(r["world"], r["seed"]) for r in rows}) == 9
    max_diff = 0.0
    for r in rows:
        w, s, method, n = int(r["world"]), int(r["seed"]), r["method"], int(r["n"])
        keys, basis, angles, values = RUN.make_world(w, s)
        qe, qp, qn = RUN.make_queries(w, s, keys, values)
        blob = RUN.make_payload(method, keys, basis, angles, values, n)
        assert len(blob) == int(r["payload_bytes"])
        assert hashlib.sha256(blob).hexdigest() == r["hash"]
        if method != "no_edit":
            path = ROOT / "artifacts" / "payloads" / f'{w}_{s}_{method}_N{n}.pt'
            assert path.read_bytes() == blob
            obj = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
            buf = io.BytesIO(); torch.save(obj, buf)
            assert buf.getvalue() == blob
        kh, vh, thr = RUN.decode(method, blob)
        m = RUN.evaluate(keys[:n], values[:n], qe[:n], qp[:n*4], qn, kh, vh, thr)
        cols = ("exact_hit_recall", "paraphrase_hit_recall", "unrelated_false_trigger_rate", "exact_value_nrmse", "paraphrase_value_nrmse", "unrelated_output_drift")
        for actual, col in zip(m, cols):
            diff = abs(actual-float(r[col])); max_diff=max(max_diff,diff)
            assert diff < 1e-12
    assert max_diff < 1e-12


def test_n64_retrieval_and_storage_gates():
    rows = list(csv.DictReader((ROOT / "RESULTS_CORE.csv").open()))
    for w in (47510, 47511, 47512):
        for s in (0, 1, 2):
            g = [r for r in rows if int(r["world"]) == w and int(r["seed"]) == s and int(r["n"]) == 64]
            by = {r["method"]:r for r in g}
            for m in ("serac", "generic_coeff", "mirror"):
                assert float(by[m]["exact_hit_recall"]) >= .99
                assert float(by[m]["paraphrase_hit_recall"]) >= .99
                assert float(by[m]["unrelated_false_trigger_rate"]) <= .01
                assert float(by[m]["paraphrase_value_nrmse"]) <= 1e-5
            assert int(by["mirror"]["payload_bytes"]) <= .8*int(by["serac"]["payload_bytes"])
