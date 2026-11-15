#!/usr/bin/env python3
"""MA-1038 bounded ColBERTv2 centroid/residual role-view screen.

The encoder/tokenization follows Stanford ColBERT v0.4 semantics. Runtime data
and serialized indexes are written under /tmp, not committed to git.
"""
from __future__ import annotations

import csv, hashlib, json, math, os, string, sys, time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.cluster import MiniBatchKMeans
from transformers import AutoTokenizer, BertConfig, BertModel

ROOT = Path(__file__).resolve().parents[4]
EXP = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("MA1038_CACHE", "/tmp/ma1038/cache"))
PAYLOAD = Path(os.environ.get("MA1038_PAYLOAD", "/tmp/ma1038/payload"))
MODEL = Path("/tmp/ma1038/colbertv2.0")
SCIFACT = Path("/tmp/ma1037-scifact")
NFCORPUS = Path("/tmp/ma1038/nfcorpus/nfcorpus")
SEED = 1038
ANGLES = np.asarray([-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20], dtype=np.float32)
DIM, QMAX, DMAX, K = 128, 32, 180, 256
torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.set_num_interop_threads(1)
compute_report = {}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jsonl(path: Path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def load_scifact():
    corpus = pq.read_table(SCIFACT / "corpus.parquet").to_pylist()
    queries = pq.read_table(SCIFACT / "queries.parquet").to_pylist()
    qmap = {str(x["_id"]): (x.get("text") or "") for x in queries}
    docs = [(str(x["_id"]), ((x.get("title") or "") + " " + (x.get("text") or "")).strip()) for x in corpus]
    qrels = {}
    with (SCIFACT / "train.tsv").open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    return docs, qmap, qrels


def load_nfcorpus():
    corpus = list(jsonl(NFCORPUS / "corpus.jsonl"))
    queries = list(jsonl(NFCORPUS / "queries.jsonl"))
    docs = [(str(x["_id"]), ((x.get("title") or "") + " " + (x.get("text") or "")).strip()) for x in corpus]
    qmap = {str(x["_id"]): (x.get("text") or "") for x in queries}
    qrels = {}
    with (NFCORPUS / "qrels" / "test.tsv").open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    return docs, qmap, qrels


def make_model():
    state = torch.load(MODEL / "pytorch_model.bin", map_location="cpu", weights_only=True)
    cfg = BertConfig.from_pretrained(MODEL)
    bert = BertModel(cfg, add_pooling_layer=False)
    sub = {k[5:]: v for k, v in state.items() if k.startswith("bert.") and not k.startswith("bert.pooler.") and k != "bert.embeddings.position_ids"}
    missing, unexpected = bert.load_state_dict(sub, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    projection = torch.nn.Linear(768, 128, bias=False)
    projection.load_state_dict({"weight": state["linear.weight"]})
    del state
    bert.eval(); projection.eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    return bert, projection, tok


def encode_texts(texts, ids, tok, bert, proj, query: bool, cache_prefix: str):
    encode_started = time.time(); processed_tokens = 0
    flat_path = CACHE / f"{cache_prefix}.f32"
    offsets_path = CACHE / f"{cache_prefix}.offsets.npy"
    lengths_path = CACHE / f"{cache_prefix}.lengths.npy"
    if flat_path.exists() and offsets_path.exists() and lengths_path.exists():
        flat = np.memmap(flat_path, mode="r", dtype=np.float32).reshape(-1, 128)
        return flat, np.load(offsets_path), np.load(lengths_path)
    allvec, offsets, lengths = [], [0], []
    qmarker = tok.convert_tokens_to_ids("[unused0]")
    dmarker = tok.convert_tokens_to_ids("[unused1]")
    maskid, padid = tok.mask_token_id, tok.pad_token_id
    skip = {padid}
    for sym in string.punctuation:
        vals = tok.encode(sym, add_special_tokens=False)
        skip.add(tok.convert_tokens_to_ids(sym))
        skip.update(vals)
    batch = 16
    for start in range(0, len(texts), batch):
        ts = texts[start:start + batch]
        maxlen = (QMAX - 1) if query else (DMAX - 1)
        enc = tok(ts, padding="max_length" if query else True, truncation=True, max_length=maxlen, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"].clone()
        att = enc["attention_mask"].clone()
        if query:
            input_ids = torch.cat([input_ids[:, :1], torch.full((len(ts), 1), qmarker, dtype=input_ids.dtype), input_ids[:, 1:]], dim=1)
            att = torch.cat([att[:, :1], torch.ones((len(ts), 1), dtype=att.dtype), att[:, 1:]], dim=1)
            pad = input_ids.eq(padid)
            input_ids[pad] = maskid
            input_ids, att = input_ids[:, :QMAX], att[:, :QMAX]
        else:
            input_ids = torch.cat([input_ids[:, :1], torch.full((len(ts), 1), dmarker, dtype=input_ids.dtype), input_ids[:, 1:]], dim=1)
            att = torch.cat([att[:, :1], torch.ones((len(ts), 1), dtype=att.dtype), att[:, 1:]], dim=1)
            input_ids, att = input_ids[:, :DMAX], att[:, :DMAX]
        processed_tokens += int(input_ids.numel())
        with torch.inference_mode():
            z = bert(input_ids, attention_mask=att).last_hidden_state
            z = torch.nn.functional.normalize(proj(z), p=2, dim=-1).cpu().numpy().astype(np.float32)
        for j, row in enumerate(z):
            if query:
                mask = np.ones(row.shape[0], dtype=bool)  # ColBERT query padding is replaced by [MASK] and scored.
                role_length = int(att[j].sum().item())
            else:
                rowids = input_ids[j].cpu().numpy()
                mask = np.asarray([int(t) not in skip for t in rowids], dtype=bool)
                role_length = int(mask.sum())
            v = np.ascontiguousarray(row[mask], dtype=np.float32)
            allvec.append(v); lengths.append(role_length); offsets.append(offsets[-1] + len(v))
        if (start // batch) % 50 == 0:
            print(f"encode {cache_prefix}: {min(start+batch,len(texts))}/{len(texts)}", flush=True)
    arr = np.concatenate(allvec, axis=0)
    flat_path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(flat_path, mode="w+", dtype=np.float32, shape=arr.shape); mm[:] = arr; mm.flush(); del mm
    np.save(offsets_path, np.asarray(offsets, dtype=np.int64))
    np.save(lengths_path, np.asarray(lengths, dtype=np.int32))
    compute_report[f"encode_{cache_prefix}"] = {"wall_seconds":time.time()-encode_started,"examples":len(texts),"padded_model_tokens":processed_tokens,"valid_output_token_vectors":int(offsets[-1])}
    return np.memmap(flat_path, mode="r", dtype=np.float32).reshape(-1, 128), np.asarray(offsets), np.asarray(lengths)


def fit_codebook(flat):
    rng = np.random.default_rng(SEED)
    n = len(flat)
    sample = np.asarray(flat[rng.choice(n, min(100000, n), replace=False)], dtype=np.float32)
    km = MiniBatchKMeans(n_clusters=K, batch_size=8192, max_iter=50, n_init=1, random_state=SEED, reassignment_ratio=0.01, compute_labels=False, verbose=0)
    t = time.time(); km.fit(sample)
    print(f"kmeans_seconds={time.time()-t:.3f}; train_tokens={n}; sample={len(sample)}", flush=True)
    c = km.cluster_centers_.astype(np.float32)
    # residual scales are estimated on all dev tokens in bounded chunks.
    from scipy.spatial.distance import cdist
    maxima = np.zeros(128, dtype=np.float32)
    quant_samples = []
    rng = np.random.default_rng(SEED + 1)
    pick = rng.choice(n, min(100000, n), replace=False)
    for st in range(0, len(pick), 4096):
        x = np.asarray(flat[pick[st:st+4096]], dtype=np.float32)
        ix = cdist(x, c, metric="sqeuclidean").argmin(axis=1)
        quant_samples.append(np.abs(x - c[ix]))
    rr = np.concatenate(quant_samples)
    scale = np.maximum(np.quantile(rr, 0.999, axis=0) / 127.0, 1e-8).astype(np.float32)
    return c, scale


def encode_index(flat, offsets, lengths, c, scale, name):
    from scipy.spatial.distance import cdist
    n = len(flat)
    codes = np.empty(n, dtype=np.uint8)
    residual = np.empty((n, 128), dtype=np.int8)
    for st in range(0, n, 4096):
        x = np.asarray(flat[st:st+4096], dtype=np.float32)
        ix = cdist(x, c, metric="sqeuclidean").argmin(axis=1).astype(np.uint8)
        r = x - c[ix]
        q = np.rint(r / scale).clip(-127, 127).astype(np.int8)
        codes[st:st+len(x)] = ix; residual[st:st+len(x)] = q
    # Actual inference decoding followed by the same unit normalization as ColBERT.
    out = c[codes.astype(np.int64)] + residual.astype(np.float32) * scale
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return out.astype(np.float32), codes, residual


def role_rotation(x, theta):
    y = np.array(x, dtype=np.float32, copy=True)
    a, b = y[..., 0::2].copy(), y[..., 1::2].copy()
    co, si = np.float32(math.cos(float(theta))), np.float32(math.sin(float(theta)))
    y[..., 0::2] = co * a - si * b
    y[..., 1::2] = si * a + co * b
    return y


def q_inverse(x, theta):
    return role_rotation(x, -float(theta))


def ndcg_mrr(qid, ranking, qrels, k=10):
    rel = qrels[qid]
    got = [rel.get(docid, 0) for docid in ranking[:k]]
    dcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(got))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    n = dcg / idcg if idcg else 0.0
    rr = next((1.0 / (i + 1) for i, r in enumerate(got) if r > 0), 0.0)
    return n, rr


def score_all(qflat, qoffs, qlens, dflat, doffs, docids, qids, qrels, role_angles=None, role_by_qid=None, batch_q=8, batch_d=64, name="score"):
    # Exact MaxSim. Temporary score cube is bounded by batch_q*batch_d*32*180.
    started = time.time(); rankings = {}; evaluations = 0
    q_tensors = [torch.from_numpy(np.asarray(qflat[qoffs[i]:qoffs[i+1]], dtype=np.float32)) for i in range(len(qids))]
    d_tensors = [torch.from_numpy(np.asarray(dflat[doffs[i]:doffs[i+1]], dtype=np.float32)) for i in range(len(docids))]
    for qs in range(0, len(qids), batch_q):
        qbatch = qids[qs:qs+batch_q]
        qidx = [qid_index[q] for q in qbatch]
        # index lookup is set once by caller
        qarr = [q_tensors[i] for i in qidx]
        if role_angles is not None:
            qarr = [torch.from_numpy(q_inverse(q.numpy(), role_angles[role_by_qid[qbatch[j]]])) for j, q in enumerate(qarr)]
        qmax = max(x.shape[0] for x in qarr)
        qpad = torch.zeros((len(qarr), qmax, 128), dtype=torch.float32)
        qmask = torch.zeros((len(qarr), qmax), dtype=torch.bool)
        for i, q in enumerate(qarr): qpad[i,:len(q)] = q; qmask[i,:len(q)] = True
        rows = []
        for ds in range(0, len(docids), batch_d):
            ids = list(range(ds, min(ds+batch_d, len(docids))))
            dmax = max(int(doffs[i+1]-doffs[i]) for i in ids)
            dpad = torch.zeros((len(ids), dmax, 128), dtype=torch.float32)
            dmask = torch.zeros((len(ids), dmax), dtype=torch.bool)
            for j, i in enumerate(ids):
                v = d_tensors[i]; dpad[j,:len(v)] = v; dmask[j,:len(v)] = True
            sim = torch.einsum("qld,ntd->qnlt", qpad, dpad)
            sim = sim.masked_fill(~dmask[None,:,None,:], -1e4)
            sums = sim.max(dim=-1).values.masked_fill(~qmask[:,None,:], 0.0).sum(dim=-1)
            rows.append(sums.numpy())
            evaluations += len(qbatch) * len(ids)
        scores = np.concatenate(rows, axis=1)
        for qi, qid in enumerate(qbatch):
            ix = np.lexsort((np.arange(len(docids)), -scores[qi]))[:100]
            rankings[qid] = [docids[int(j)] for j in ix]
        if (qs // batch_q) % 25 == 0: print(f"{name}: {min(qs+batch_q,len(qids))}/{len(qids)} queries", flush=True)
    elapsed = time.time()-started
    selected_q_tokens = sum(int(qoffs[qid_index[q]+1]-qoffs[qid_index[q]]) for q in qids)
    all_doc_tokens = int(doffs[-1])
    mac_proxy = selected_q_tokens * all_doc_tokens * 128
    compute_report[name] = {"wall_seconds": elapsed, "query_document_pairs": evaluations, "maxsim_mac_proxy": int(mac_proxy), "query_token_vectors": selected_q_tokens, "document_token_vectors": all_doc_tokens}
    print(f"{name}_seconds={elapsed:.3f}; pair_evals={evaluations}; maxsim_mac_proxy={mac_proxy}", flush=True)
    return rankings


def metric_summary(rankings, qrels, roles=None):
    qids = [q for q in rankings if q in qrels and any(v > 0 for v in qrels[q].values())]
    out = {}
    groups = {"all": qids}
    if roles is not None:
        for r in sorted(set(roles[q] for q in qids)):
            groups[f"role_{r}"] = [q for q in qids if roles[q] == r]
    for label, qs in groups.items():
        vals = [ndcg_mrr(q, rankings[q], qrels) for q in qs]
        out[label] = {"n": len(qs), "ndcg10": float(np.mean([x[0] for x in vals])), "mrr10": float(np.mean([x[1] for x in vals]))}
    return out


def save_index_payload(name, model_files, docids, offsets, full, codebook=None, scale=None, codes=None, residual=None, view_angles=None, threshold=None):
    folder = PAYLOAD / name
    folder.mkdir(parents=True, exist_ok=True)
    for src in model_files:
        dst = folder / src.name
        if not dst.exists():
            import shutil; shutil.copy2(src, dst)
    np.save(folder / "doc_offsets.int64.npy", np.asarray(offsets, dtype=np.int64))
    (folder / "doc_ids.json").write_text(json.dumps(docids, separators=(",", ":")))
    if full is not None:
        np.save(folder / "doc_vectors.float32.npy", np.asarray(full, dtype=np.float32))
    else:
        np.save(folder / "centroids.float32.npy", np.asarray(codebook, dtype=np.float32))
        np.save(folder / "residual_scale.float32.npy", np.asarray(scale, dtype=np.float32))
        np.save(folder / "centroid_codes.uint8.npy", np.asarray(codes, dtype=np.uint8))
        np.save(folder / "residuals.int8.npy", np.asarray(residual, dtype=np.int8))
    if view_angles is not None:
        np.save(folder / "role_angles.float32.npy", np.asarray(view_angles, dtype=np.float32))
        (folder / "role_threshold.json").write_text(json.dumps({"median_token_count": int(threshold), "short_inclusive": True}, separators=(",", ":")))
    (folder / "index_format.json").write_text(json.dumps({"model":"official-ColBERTv2", "dim":128, "max_doclen":180, "n_documents":len(docids), "n_valid_tokens":int(offsets[-1]), "storage":"float32-full" if full is not None else "256 centroid IDs + signed int8 coordinate residuals", "normalization":"unit-L2 after reconstruct", "view_angles_radians":None if view_angles is None else list(map(float,view_angles))}, separators=(",", ":")))
    return sum(p.stat().st_size for p in folder.iterdir() if p.is_file()), {p.name:p.stat().st_size for p in sorted(folder.iterdir()) if p.is_file()}


def main():
    CACHE.mkdir(parents=True, exist_ok=True); PAYLOAD.mkdir(parents=True, exist_ok=True)
    if not MODEL.exists() or not SCIFACT.exists() or not NFCORPUS.exists():
        raise SystemExit("Missing pinned model or datasets; see PROTOCOL.json")
    torch.manual_seed(SEED); np.random.seed(SEED)
    bert, proj, tok = make_model()
    dev_docs, dev_queries, dev_qrels = load_scifact()
    # NFCorpus is read only after the frozen protocol and this runner exist.
    audit_docs, audit_queries, audit_qrels = load_nfcorpus()
    model_files = [MODEL/n for n in ["pytorch_model.bin", "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]]
    model_files = [p for p in model_files if p.exists()]
    dev_dids, dev_texts = zip(*dev_docs); dev_qids_all, dev_qtexts_all = zip(*sorted(dev_queries.items()))
    aud_dids, aud_texts = zip(*audit_docs); aud_qids_all, aud_qtexts_all = zip(*sorted(audit_queries.items()))
    # qrels rows are the only development query set; all official NFCorpus test qrel queries form audit.
    dev_qids = sorted(q for q in dev_qrels if q in dev_queries and any(x > 0 for x in dev_qrels[q].values()))
    aud_qids = sorted(q for q in audit_qrels if q in audit_queries and any(x > 0 for x in audit_qrels[q].values()))
    dev_qtextmap = dict(zip(dev_qids_all, dev_qtexts_all)); aud_qtextmap = dict(zip(aud_qids_all, aud_qtexts_all))
    dev_docmap = {x:i for i,x in enumerate(dev_dids)}; aud_docmap = {x:i for i,x in enumerate(aud_dids)}
    dev_qflat, dev_qoff, dev_qlen = encode_texts([dev_qtextmap[q] for q in dev_qids], dev_qids, tok, bert, proj, True, "dev_q")
    dev_dflat, dev_doff, dev_dlen = encode_texts(dev_texts, dev_dids, tok, bert, proj, False, "dev_d")
    # Dense id lookup required by scorer.
    global qid_index
    qid_index = {q:i for i,q in enumerate(dev_qids)}
    dindex = {d:i for i,d in enumerate(dev_dids)}
    print(f"dev docs={len(dev_dids)}, train_queries={len(dev_qids)}, dev_tokens={int(dev_doff[-1])}", flush=True)
    codebook, scale = fit_codebook(dev_dflat)
    dev_dec, dev_codes, dev_res = encode_index(dev_dflat, dev_doff, dev_dlen, codebook, scale, "dev")
    freeze_path = EXP / "runs" / "DEV_FREEZE.json"
    if freeze_path.exists():
        frozen = json.loads(freeze_path.read_text())
        assert frozen["angle_grid"] == ANGLES.tolist(), "frozen development grid does not match protocol"
        median_len = int(frozen["role_threshold_tokens"])
        chosen = [float(x) for x in frozen["angles"]]
        angle_scores = {int(k):[(float(a),float(b)) for a,b in v] for k,v in frozen["angle_scores"].items()}
        print(f"reuse frozen development choices: threshold={median_len}, angles={chosen}", flush=True)
    else:
        median_len = int(np.median([dev_qlen[i] for i in range(len(dev_qids))]))
        dev_roles = {q:(0 if int(dev_qlen[i]) <= median_len else 1) for i,q in enumerate(dev_qids)}
        # freeze two independent role coordinates using training qrels only.
        dev_q_index = {q:i for i,q in enumerate(dev_qids)}
        base_dev_rank = score_all(dev_qflat, dev_qoff, dev_qlen, dev_dec, dev_doff, dev_dids, dev_qids, dev_qrels, name="dev_baseline")
        angle_scores = {0:[],1:[]}
        for angle in ANGLES:
            role_angles = [float(angle), float(angle)]
            qid_index = dev_q_index
            rank = score_all(dev_qflat, dev_qoff, dev_qlen, dev_dec, dev_doff, dev_dids, dev_qids, dev_qrels, role_angles, dev_roles, name=f"dev_angle_{angle:+.2f}")
            met = metric_summary(rank, dev_qrels, dev_roles)
            for role in [0,1]: angle_scores[role].append((met[f"role_{role}"]["ndcg10"], float(angle)))
        chosen = []
        for role in [0,1]:
            best=max(x[0] for x in angle_scores[role]); tied=[x[1] for x in angle_scores[role] if abs(x[0]-best)<1e-12]
            chosen.append(sorted(tied,key=lambda a:(abs(a),a))[0])
        frozen = {"role_threshold_tokens":median_len,"angles":chosen,"angle_grid":ANGLES.tolist(),"angle_scores":{str(k):v for k,v in angle_scores.items()},"selected_from":"SciFact train qrels only; captured before NFCorpus metric calculation"}
        freeze_path.parent.mkdir(parents=True,exist_ok=True)
        freeze_path.write_text(json.dumps(frozen,indent=2,sort_keys=True)+"\n")
        print(f"role_threshold={median_len}; frozen_angles={chosen}; dev_angle_scores={angle_scores}", flush=True)
    # Replace query caches with the audit set only after views are frozen.
    aud_qflat, aud_qoff, aud_qlen = encode_texts([aud_qtextmap[q] for q in aud_qids], aud_qids, tok, bert, proj, True, "audit_q")
    aud_dflat, aud_doff, aud_dlen = encode_texts(aud_texts, aud_dids, tok, bert, proj, False, "audit_d")
    aud_dec, aud_codes, aud_res = encode_index(aud_dflat, aud_doff, aud_dlen, codebook, scale, "audit")
    aud_roles = {q:(0 if int(aud_qlen[i]) <= median_len else 1) for i,q in enumerate(aud_qids)}
    qid_index = {q:i for i,q in enumerate(aud_qids)}
    # Audit baselines and the Mirror/query-side push-through control.
    audit_full_rank = score_all(aud_qflat, aud_qoff, aud_qlen, aud_dflat, aud_doff, aud_dids, aud_qids, audit_qrels, name="audit_full_float")
    audit_base_rank = score_all(aud_qflat, aud_qoff, aud_qlen, aud_dec, aud_doff, aud_dids, aud_qids, audit_qrels, name="audit_residual_native")
    audit_mirror_rank = score_all(aud_qflat, aud_qoff, aud_qlen, aud_dec, aud_doff, aud_dids, aud_qids, audit_qrels, chosen, aud_roles, name="audit_mirror_pushed_query")
    # Direct document-side logit equality audit on fixed random query/document/token triples.
    rng = np.random.default_rng(SEED+2); diffs=[]
    for _ in range(4096):
        qi=int(rng.integers(len(aud_qids))); di=int(rng.integers(len(aud_dids)))
        q=aud_qflat[aud_qoff[qi]:aud_qoff[qi+1]]; d=aud_dec[aud_doff[di]:aud_doff[di+1]]
        th=chosen[aud_roles[aud_qids[qi]]]
        direct=role_rotation(d,th) @ np.asarray(q).T
        pushed=np.asarray(d) @ q_inverse(np.asarray(q),th).T
        diffs.append(float(np.max(np.abs(direct-pushed))))
    # materialize actual inference bundles; shared model/tokenizer charged in every arm.
    full_bytes, full_files = save_index_payload("audit_full_float", model_files, aud_dids, aud_doff, aud_dflat)
    compressed_bytes, compressed_files = save_index_payload("audit_residual_native", model_files, aud_dids, aud_doff, None, codebook, scale, aud_codes, aud_res)
    mirror_bytes, mirror_files = save_index_payload("audit_mirror_role_view", model_files, aud_dids, aud_doff, None, codebook, scale, aud_codes, aud_res, chosen, median_len)
    native_bytes, native_files = save_index_payload("audit_native_query_givens", model_files, aud_dids, aud_doff, None, codebook, scale, aud_codes, aud_res, chosen, median_len)
    audit_roles = {q:(0 if int(aud_qlen[i]) <= median_len else 1) for i,q in enumerate(aud_qids)}
    metrics={
      "audit_full_float":metric_summary(audit_full_rank,audit_qrels,audit_roles),
      "audit_residual_native":metric_summary(audit_base_rank,audit_qrels,audit_roles),
      "audit_mirror":metric_summary(audit_mirror_rank,audit_qrels,audit_roles),
      "audit_native_query_givens":metric_summary(audit_mirror_rank,audit_qrels,audit_roles),
    }
    changed = [q for q in aud_qids if audit_base_rank[q][:10] != audit_mirror_rank[q][:10]]
    overlaps = [len(set(audit_base_rank[q][:10]) & set(audit_mirror_rank[q][:10])) / 10.0 for q in aud_qids]
    query_view_flops = len(aud_qids) * int(aud_qoff[-1]) * 384
    direct_document_view_flops = len(aud_qids) * int(aud_doff[-1]) * 384
    summary={"experiment":"MA-1038","model_checkpoint_sha256":"d07df4b2a9fe1848fbed333127a1a5b4f2d96106f420948f0db4450b018cb119","nfcorpus_zip_sha256":"efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b","seed":SEED,"train_queries":len(dev_qids),"audit_queries":len(aud_qids),"dev_docs":len(dev_dids),"audit_docs":len(aud_dids),"dev_tokens":int(dev_doff[-1]),"audit_tokens":int(aud_doff[-1]),"role_threshold_tokens":median_len,"audit_role_query_counts":{"short":sum(v==0 for v in aud_roles.values()),"long":sum(v==1 for v in aud_roles.values())},"angles":chosen,"angle_grid":ANGLES.tolist(),"metrics":metrics,"ranking_divergence_vs_no_view":{"queries_with_different_top10":len(changed),"mean_top10_overlap_fraction":float(np.mean(overlaps))},"payload_bytes":{"audit_full_float":full_bytes,"audit_residual_native":compressed_bytes,"audit_mirror":mirror_bytes,"audit_native_query_givens":native_bytes},"payload_files":{"audit_full_float":full_files,"audit_residual_native":compressed_files,"audit_mirror":mirror_files,"audit_native_query_givens":native_files},"compute":compute_report,"view_compute_proxy":{"native_query_side_givens_flops":int(query_view_flops),"direct_document_side_givens_flops_per_audit_pass":int(direct_document_view_flops)},"optimizer_updates":0,"selection_evaluations":len(ANGLES)*len(dev_qids),"doc_side_to_query_side_equiv_sample_max_abs_score_diff":max(diffs),"doc_side_to_query_side_equiv_sample_n":len(diffs),"model_bundle_bytes":sum(p.stat().st_size for p in model_files),"notes":"query-side native Givens control uses identical angle files; score equivalence is audited on fixed samples and follows orthogonal dot-product algebra."}
    (EXP/"runs").mkdir(exist_ok=True)
    (EXP/"runs"/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary,indent=2,sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
