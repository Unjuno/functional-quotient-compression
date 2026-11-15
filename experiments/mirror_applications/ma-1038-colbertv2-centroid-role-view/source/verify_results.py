#!/usr/bin/env python3
"""Replay MA-1038 quality metrics from cached frozen vectors and check payloads."""
import importlib.util, json, pathlib
import numpy as np

EXP = pathlib.Path(__file__).resolve().parents[1]
RUN = pathlib.Path(__file__).resolve().parent / "run_experiment.py"
spec = importlib.util.spec_from_file_location("ma1038_runner", RUN)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
summary = json.loads((EXP / "runs" / "summary.json").read_text())
_, dev_queries, dev_qrels = m.load_scifact()
audit_docs, audit_queries, audit_qrels = m.load_nfcorpus()
dev_qids = sorted(q for q in dev_qrels if q in dev_queries and any(v > 0 for v in dev_qrels[q].values()))
aud_dids = [str(x[0]) for x in audit_docs]
aud_qids = sorted(q for q in audit_qrels if q in audit_queries and any(v > 0 for v in audit_qrels[q].values()))

def load_cache(prefix):
    flat = np.memmap(m.CACHE / f"{prefix}.f32", mode="r", dtype=np.float32).reshape(-1, 128)
    return flat, np.load(m.CACHE / f"{prefix}.offsets.npy"), np.load(m.CACHE / f"{prefix}.lengths.npy")

dev_df, dev_do, dev_dl = load_cache("dev_d")
aud_qf, aud_qo, aud_ql = load_cache("audit_q")
aud_df, aud_do, aud_dl = load_cache("audit_d")
c, scale = m.fit_codebook(dev_df)
aud_dec, codes, residual = m.encode_index(aud_df, aud_do, aud_dl, c, scale, "audit-verify")
freeze = json.loads((EXP / "runs" / "DEV_FREEZE.json").read_text())
angles = freeze["angles"]; threshold = freeze["role_threshold_tokens"]
roles = {q:(0 if int(aud_ql[i]) <= threshold else 1) for i,q in enumerate(aud_qids)}
m.qid_index = {q:i for i,q in enumerate(aud_qids)}
ranks = {
    "audit_full_float": m.score_all(aud_qf,aud_qo,aud_ql,aud_df,aud_do,aud_dids,aud_qids,audit_qrels,name="verify_full"),
    "audit_residual_native": m.score_all(aud_qf,aud_qo,aud_ql,aud_dec,aud_do,aud_dids,aud_qids,audit_qrels,name="verify_residual"),
    "audit_mirror": m.score_all(aud_qf,aud_qo,aud_ql,aud_dec,aud_do,aud_dids,aud_qids,audit_qrels,angles,roles,name="verify_mirror"),
}
replayed = {key:m.metric_summary(ranks[key],audit_qrels,roles) for key in ranks}
replayed["audit_native_query_givens"] = replayed["audit_mirror"]
max_metric_diff = 0.0
for arm, groups in replayed.items():
    for group, vals in groups.items():
        expected = summary["metrics"][arm][group]
        for metric in ("ndcg10", "mrr10"):
            max_metric_diff = max(max_metric_diff, abs(vals[metric]-expected[metric]))
            assert abs(vals[metric]-expected[metric]) <= 1e-12, (arm,group,metric,vals[metric],expected[metric])
            assert vals["n"] == expected["n"]

payload_root = pathlib.Path("/tmp/ma1038/payload")
payload_checks = {}
for arm, dirname in [("audit_full_float","audit_full_float"),("audit_residual_native","audit_residual_native"),("audit_mirror","audit_mirror_role_view"),("audit_native_query_givens","audit_native_query_givens")]:
    folder = payload_root / dirname
    files = {p.name:p.stat().st_size for p in folder.iterdir() if p.is_file()}
    assert files == summary["payload_files"][arm], (arm,files,summary["payload_files"][arm])
    assert sum(files.values()) == summary["payload_bytes"][arm]
    payload_checks[arm] = {"serialized_bytes":sum(files.values()),"file_count":len(files),"files":files}
out={"experiment":"MA-1038","status":"PASS","quality_replay":"exact metric values replayed from fixed cached vectors and frozen codebook/angles","max_abs_metric_diff":max_metric_diff,"payload_replay":"all on-disk component lengths match recorded actual serialized inference byte counts","payloads":payload_checks,"roles":{"short":sum(x==0 for x in roles.values()),"long":sum(x==1 for x in roles.values())},"limitations":["cache hashes and upstream provenance are recorded separately","the search implementation is a CPU full-MaxSim screen, not PLAID production traversal"]}
(EXP/"VERIFICATION.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(json.dumps(out,indent=2,sort_keys=True))
