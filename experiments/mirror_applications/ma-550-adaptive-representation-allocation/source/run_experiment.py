#!/usr/bin/env python3
"""MA-550 deterministic mixed output-delta representation screen."""
from __future__ import annotations
import argparse, hashlib, json, struct, time
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve

MODEL_SHA = "3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd"
REVISION = "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
WIDTH, N_FUNCTIONS, N_SPARSE, N_SUPPORT, N_HELDOUT = 512, 16, 8, 8, 8
NOISE_SD, CODE_SD, TOL = 1e-4, 0.05, 0.01


def read_safetensor(path: Path, tensor_name: str) -> np.ndarray:
    """Read one contiguous F32 safetensors tensor without loading other tensors."""
    raw = path.read_bytes()
    header_len = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + header_len])
    item = header[tensor_name]
    if item["dtype"] not in ("F32", "F16"):
        raise ValueError(f"expected F32/F16, got {item['dtype']}")
    start, end = item["data_offsets"]
    data_start = 8 + header_len
    dtype = "<f4" if item["dtype"] == "F32" else "<f2"
    return np.frombuffer(raw, dtype=dtype, count=(end - start) // np.dtype(dtype).itemsize,
                         offset=data_start + start).reshape(item["shape"]).astype(np.float32)


def world(seed: int, w: np.ndarray):
    rng = np.random.default_rng(seed)
    vocab, d = w.shape
    ids = rng.choice(vocab, size=N_SPARSE, replace=False)
    targets = []
    kinds = []
    for i in range(N_FUNCTIONS):
        if i < N_SPARSE:
            q = np.zeros(vocab, np.float32)
            q[ids[i]] = 2.0
            kinds.append("sparse")
        else:
            m = rng.normal(0, CODE_SD, size=d).astype(np.float32)
            q = w @ m
            kinds.append("activation")
        support = q[None, :] + rng.normal(0, NOISE_SD, (N_SUPPORT, vocab)).astype(np.float32)
        held = q[None, :] + rng.normal(0, NOISE_SD, (N_HELDOUT, vocab)).astype(np.float32)
        targets.append((q, support, held))
    return kinds, targets


def rel_rmse(pred, target):
    return float(np.linalg.norm(pred - target) / max(np.linalg.norm(target), 1e-12))


def _npz_bytes(path: Path, arrays: dict) -> int:
    np.savez(path, **arrays)
    return path.stat().st_size


def run(seed: int, split: str, model_dir: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.safetensors"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != MODEL_SHA:
        raise ValueError("pinned model hash mismatch")
    t0 = time.perf_counter()
    sd = read_safetensor(model_path, "embed_out.weight")
    # GPT-NeoX checkpoint output matrix is expected to be [vocab, hidden].
    if sd.shape[1] != WIDTH:
        raise ValueError(f"unexpected output projection shape {sd.shape}")
    w = sd.astype(np.float64)
    w_sha = hashlib.sha256(np.ascontiguousarray(sd).tobytes()).hexdigest()
    load_s = time.perf_counter() - t0
    kinds, records = world(seed, sd)

    # Reuse the fixed paid native basis W; factorization is data-independent setup.
    t0 = time.perf_counter()
    gram = w.T @ w
    chol = cho_factor(gram, lower=True, check_finite=False)
    factor_s = time.perf_counter() - t0
    estimates = []
    fit_s = 0.0
    for q, support, held in records:
        y = support.mean(axis=0, dtype=np.float64)
        t0 = time.perf_counter()
        code = cho_solve(chol, w.T @ y, check_finite=False)
        fit_s += time.perf_counter() - t0
        estimates.append((y.astype(np.float32), code.astype(np.float32), held.mean(axis=0).astype(np.float32)))

    methods = ("weight_dense", "activation_code", "weight_sparse", "adaptive")
    outputs = {m: [] for m in methods}
    selections = []
    # Build comparable full-world inference payloads, with identical task IDs/metadata.
    dense_est = np.stack([x[0] for x in estimates])
    codes = np.stack([x[1] for x in estimates])
    sparse_rows, sparse_cols, sparse_vals = [], [], []
    sparse_decoded = []
    for i, (y, code, _) in enumerate(estimates):
        j = int(np.argmax(np.abs(y)))
        val = float(y[j])
        z = np.zeros_like(y); z[j] = val
        sparse_rows.append(i); sparse_cols.append(j); sparse_vals.append(val)
        sparse_decoded.append(z)
    sparse_decoded = np.stack(sparse_decoded).astype(np.float32)
    dense_payload = {"delta_logits": dense_est, "function_ids": np.arange(N_FUNCTIONS, dtype=np.int16),
                     "seed": np.array([seed], np.int64), "model_sha256": np.frombuffer(MODEL_SHA.encode(), dtype="S64"),
                     "schema": np.array([550, 1], np.int32)}
    activation_payload = {"activation_codes": codes, "function_ids": np.arange(N_FUNCTIONS, dtype=np.int16),
                          "seed": np.array([seed], np.int64), "model_sha256": np.frombuffer(MODEL_SHA.encode(), dtype="S64"),
                          "output_basis_sha256": np.frombuffer(w_sha.encode(), dtype="S64"), "schema": np.array([550, 2], np.int32)}
    sparse_payload = {"function_ids": np.asarray(sparse_rows, np.int16), "vocab_indices": np.asarray(sparse_cols, np.int32),
                      "values": np.asarray(sparse_vals, np.float32), "seed": np.array([seed], np.int64),
                      "model_sha256": np.frombuffer(MODEL_SHA.encode(), dtype="S64"), "schema": np.array([550, 3], np.int32)}
    byte_counts = {}
    byte_counts["weight_dense"] = _npz_bytes(out / "weight_dense.npz", dense_payload)
    byte_counts["activation_code"] = _npz_bytes(out / "activation_code.npz", activation_payload)
    byte_counts["weight_sparse"] = _npz_bytes(out / "weight_sparse.npz", sparse_payload)

    for i, ((q, support, held), (y, code, held_mean)) in enumerate(zip(records, estimates)):
        dense = y
        activation = sd.astype(np.float64) @ code.astype(np.float64)
        sparse = sparse_decoded[i]
        candidates = {"weight_dense": dense, "activation_code": activation, "weight_sparse": sparse}
        eligible = []
        support_err = {m: rel_rmse(pred, support.mean(axis=0)) for m, pred in candidates.items()}
        for m in ("weight_dense", "activation_code", "weight_sparse"):
            if support_err[m] <= TOL:
                # Per-function actual encoded size; adaptive pays a family tag/index overhead.
                sizes = {"weight_dense": dense.nbytes, "activation_code": code.nbytes,
                         "weight_sparse": 4 + 4 + 2}
                eligible.append((sizes[m] + 1, m))
        chosen = min(eligible)[1] if eligible else "weight_dense"
        selections.append({"function_id": i, "generated_family": kinds[i], "selected_family": chosen,
                           "support_relative_rmse": support_err[chosen]})
        for m in ("weight_dense", "activation_code", "weight_sparse"):
            pred = candidates[m]
            outputs[m].append({"support": support_err[m], "heldout": rel_rmse(pred, q), "pred": pred})
        outputs["adaptive"].append({"support": support_err[chosen], "heldout": rel_rmse(candidates[chosen], q), "pred": candidates[chosen], "family": chosen})

    adaptive_pred = np.stack([x["pred"] for x in outputs["adaptive"]]).astype(np.float32)
    tags = np.asarray([methods.index(x["family"]) for x in outputs["adaptive"]], dtype=np.int8)
    adaptive_payload = {"delta_by_function": adaptive_pred, "family_tags": tags,
                        "function_ids": np.arange(N_FUNCTIONS, dtype=np.int16), "seed": np.array([seed], np.int64),
                        "model_sha256": np.frombuffer(MODEL_SHA.encode(), dtype="S64"),
                        "schema": np.array([550, 4], np.int32)}
    # Store only each selected native code in adaptive inference state, in a genuinely ragged payload.
    # Dense arrays above are a decoder-friendly audit dump; the charged payload below uses packed entries.
    packed = {"family_tags": tags, "function_ids": np.arange(N_FUNCTIONS, dtype=np.int16),
              "seed": np.array([seed], np.int64), "model_sha256": np.frombuffer(MODEL_SHA.encode(), dtype="S64"),
              "schema": np.array([550, 5], np.int32)}
    for fam, key in (("weight_dense", "dense"), ("activation_code", "activation"), ("weight_sparse", "sparse")):
        ix = np.asarray([i for i, x in enumerate(outputs["adaptive"]) if x["family"] == fam], dtype=np.int16)
        packed[key + "_ids"] = ix
        if fam == "weight_dense": packed[key + "_values"] = dense_est[ix]
        elif fam == "activation_code": packed[key + "_values"] = codes[ix]
        else:
            packed[key + "_indices"] = np.asarray([sparse_cols[i] for i in ix], np.int32)
            packed[key + "_values"] = np.asarray([sparse_vals[i] for i in ix], np.float32)
    byte_counts["adaptive"] = _npz_bytes(out / "adaptive.npz", packed)

    metrics = {}
    for method in methods:
        vals = outputs[method]
        metrics[method] = {"payload_bytes": byte_counts.get(method, 0),
                           "support_relative_rmse": float(np.mean([x["support"] for x in vals])),
                           "max_support_relative_rmse": float(np.max([x["support"] for x in vals])),
                           "heldout_relative_rmse": float(np.mean([x["heldout"] for x in vals])),
                           "max_heldout_relative_rmse": float(np.max([x["heldout"] for x in vals]))}
    metrics["weight_sparse"]["payload_bytes"] = byte_counts["weight_sparse"]
    report = {"experiment_id": "MA-550", "seed": seed, "split": split, "model_revision": REVISION,
              "model_sha256": MODEL_SHA, "output_weight_sha256": w_sha, "vocab_size": int(w.shape[0]),
              "width": WIDTH, "sparse_functions": N_SPARSE, "activation_functions": N_FUNCTIONS-N_SPARSE,
              "support_observations": N_SUPPORT, "heldout_observations": N_HELDOUT,
              "selection": selections, "methods": metrics,
              "compute": {"input_examples": N_FUNCTIONS*(N_SUPPORT+N_HELDOUT), "optimizer_updates": 0,
                          "shared_gram_flops_proxy": int(2*w.shape[0]*WIDTH*WIDTH),
                          "per_function_code_fit_flops_proxy": int(N_FUNCTIONS*(2*w.shape[0]*WIDTH+2*WIDTH*WIDTH)),
                          "model_load_seconds": load_s, "basis_factor_seconds": factor_s,
                          "code_fit_seconds": fit_s, "wall_seconds": time.perf_counter()-t0},
              "quality_gates": {m: bool(metrics[m]["max_support_relative_rmse"] <= TOL and metrics[m]["max_heldout_relative_rmse"] <= TOL) for m in methods}}
    (out / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"seed": seed, "split": split, "methods": metrics, "family_counts": {f: sum(x['family']==f for x in outputs['adaptive']) for f in ("weight_dense","activation_code","weight_sparse")}}, indent=2))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", choices=("dev", "fresh"), required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a=ap.parse_args(); run(a.seed, a.split, a.model_dir, a.out)


if __name__ == "__main__": main()
