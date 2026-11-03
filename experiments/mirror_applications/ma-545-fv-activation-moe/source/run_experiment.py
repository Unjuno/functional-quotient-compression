"""MA-545 routed residual function-vector bank on pinned Pythia-70M."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
BASE_SOURCE = REPO / "experiments/mirror_applications/ma-516-function-vector-mirror-compression/source/run_screen.py"
spec = importlib.util.spec_from_file_location("ma516_source", BASE_SOURCE)
ma516 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ma516)

D = 512
TASKS = ma516.TASKS
N_TASKS = len(TASKS)
SEEDS = {"dev": (54501, 54502), "fresh": (54511, 54512, 54513)}
METHODS = ("no_intervention", "shared_mean_fv", "oracle_task_fv", "routed_fv_top1",
           "routed_fv_top2", "oracle_weight_bias", "routed_weight_bias_top1")


def split_task(task_id: int, seed: int):
    rng = np.random.default_rng(seed + 1009 * task_id)
    ids = rng.permutation(16)
    pairs = TASKS[task_id][1]
    return ([pairs[i] for i in ids[:8]], [pairs[i] for i in ids[8:12]], [pairs[i] for i in ids[12:]])


def make_splits(seed: int):
    return [split_task(tid, seed) for tid in range(N_TASKS)]


def prompt(query, demos=()):
    return ma516.prompt(query, demos)


def extract_banks(model, tok, torch_module, splits):
    vectors, route_prototypes, manifest = [], [], []
    counters = {"fv_forward_calls": 0, "router_forward_calls": 0, "support_input_tokens": 0,
                "calibration_input_tokens": 0}
    for task_id, (support, calibration, audit) in enumerate(splits):
        deltas = []
        for x, _ in support:
            demos = [pair for pair in support if pair[0] != x]
            full, full_tokens = ma516.get_activation(model, tok, torch_module, prompt(x, demos))
            base, base_tokens = ma516.get_activation(model, tok, torch_module, prompt(x))
            deltas.append(full - base)
            counters["fv_forward_calls"] += 2
            counters["support_input_tokens"] += full_tokens + base_tokens
        vectors.append(np.mean(deltas, axis=0, dtype=np.float64).astype(np.float32))
        router_features = []
        for x, _ in calibration:
            feature, tokens = ma516.get_activation(model, tok, torch_module, prompt(x, support))
            router_features.append(feature)
            counters["router_forward_calls"] += 1
            counters["calibration_input_tokens"] += tokens
        centroid = np.mean(router_features, axis=0, dtype=np.float64).astype(np.float32)
        norm = float(np.linalg.norm(centroid))
        route_prototypes.append(centroid / max(norm, 1e-12))
        manifest.append({"task_id": task_id, "name": TASKS[task_id][0], "support": support,
                         "router_calibration": calibration, "audit": audit})
    return np.stack(vectors), np.stack(route_prototypes), manifest, counters


def route(query_feature: np.ndarray, prototypes: np.ndarray, vectors: np.ndarray, temperature=0.05):
    q = query_feature / max(float(np.linalg.norm(query_feature)), 1e-12)
    sims = q @ prototypes.T
    order = np.argsort(-sims, kind="stable")
    top1 = int(order[0])
    top2 = order[:2]
    logits = sims[top2] / temperature
    weights = np.exp(logits - np.max(logits)); weights = weights / weights.sum()
    mixed = (vectors[top2] * weights[:, None]).sum(axis=0).astype(np.float32)
    return top1, int(top2[0]), int(top2[1]), weights.astype(np.float32), mixed, sims


def _candidate_inputs(tok, torch_module, query, candidates):
    prefix = ma516.enc(tok, prompt(query))
    seqs, targets = [], []
    for candidate in candidates:
        full = ma516.enc(tok, prompt(query) + " " + candidate)
        if full[:len(prefix)] != prefix:
            raise ValueError("candidate tokenization changed the frozen prompt prefix")
        seqs.append(full); targets.append(full[len(prefix):])
    maxlen = max(map(len, seqs))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch_module.full((len(seqs), maxlen), pad, dtype=torch_module.long)
    mask = torch_module.zeros_like(ids)
    for i, seq in enumerate(seqs):
        ids[i, :len(seq)] = torch_module.tensor(seq); mask[i, :len(seq)] = 1
    return ids, mask, prefix, targets


def activation_logits(model, torch_module, ids, mask, pos, vec):
    return ma516.hook_forward(model, torch_module, ids, mask, pos, vec)


def weight_bias_logits(model, torch_module, ids, mask, pos, vec):
    """Select a task-specific MLP output-bias delta at the queried position."""
    layer = model.gpt_neox.layers[ma516.HOOK_LAYER]
    module = layer.mlp.dense_4h_to_h
    delta = torch_module.as_tensor(vec, dtype=torch_module.float32)
    def add_bias(module, args, out):
        changed = out.clone()
        changed[:, pos, :] += delta
        return changed
    handle = module.register_forward_hook(add_bias)
    try:
        with torch_module.no_grad():
            output = model(input_ids=ids, attention_mask=mask, use_cache=False)
        return output.logits
    finally:
        handle.remove()


def score_query(model, tok, torch_module, query, candidates, vec=None, mode="activation"):
    ids, mask, prefix, targets = _candidate_inputs(tok, torch_module, query, candidates)
    pos = len(prefix) - 1
    if mode == "none":
        with torch_module.no_grad():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    elif mode == "activation":
        logits = activation_logits(model, torch_module, ids, mask, pos, vec)
    elif mode == "weight_bias":
        logits = weight_bias_logits(model, torch_module, ids, mask, pos, vec)
    else:
        raise ValueError(mode)
    lp = torch_module.log_softmax(logits, dim=-1)
    scores = []
    for row, answer_tokens in enumerate(targets):
        total = 0.0
        for j, token_id in enumerate(answer_tokens):
            total += float(lp[row, len(prefix) + j - 1, token_id].item())
        scores.append(total)
    return np.asarray(scores, dtype=np.float64), len(candidates), int(mask.sum().item())


def evaluate(model, tok, torch_module, vectors, prototypes, manifest):
    rows = {m: [] for m in METHODS}
    route_correct = route_top2_contains = 0
    alias_differences = []
    total_candidates = total_input_tokens = model_calls = 0
    t0 = time.perf_counter()
    for task in manifest:
        tid = task["task_id"]
        candidates = sorted({y for _, y in TASKS[tid][1]})
        for query, gold_answer in task["audit"]:
            feature, _ = ma516.get_activation(model, tok, torch_module, prompt(query, task["support"]))
            top1, top2a, top2b, weights, mixed, sims = route(feature, prototypes, vectors)
            route_correct += int(top1 == tid)
            route_top2_contains += int(tid in (top2a, top2b))
            vectors_for = {
                "no_intervention": (None, "none"),
                "shared_mean_fv": (vectors.mean(axis=0), "activation"),
                "oracle_task_fv": (vectors[tid], "activation"),
                "routed_fv_top1": (vectors[top1], "activation"),
                "routed_fv_top2": (mixed, "activation"),
                "oracle_weight_bias": (vectors[tid], "weight_bias"),
                "routed_weight_bias_top1": (vectors[top1], "weight_bias"),
            }
            per_query_logits = {}
            for method, (vec, mode) in vectors_for.items():
                scores, calls, tokens = score_query(model, tok, torch_module, query, candidates, vec, mode)
                index = candidates.index(gold_answer)
                correct = int(int(np.argmax(scores)) == index)
                negatives = np.delete(scores, index)
                gold_lp = float(scores[index])
                rows[method].append({"task_id": tid, "correct": correct, "gold_logprob": gold_lp,
                                     "margin": float(gold_lp - np.max(negatives)),
                                     "predicted_task_id": None if method in ("no_intervention", "shared_mean_fv", "oracle_task_fv", "oracle_weight_bias") else (top1 if method.endswith("top1") else None)})
                per_query_logits[method] = scores
                total_candidates += calls; total_input_tokens += tokens; model_calls += 1
            alias_differences.append(float(np.max(np.abs(per_query_logits["oracle_task_fv"] - per_query_logits["oracle_weight_bias"]))))
    inference_seconds = time.perf_counter() - t0
    summary = {}
    for method, observations in rows.items():
        summary[method] = {
            "heldout_accuracy": float(np.mean([r["correct"] for r in observations])),
            "mean_gold_logprob": float(np.mean([r["gold_logprob"] for r in observations])),
            "mean_gold_vs_best_negative_margin": float(np.mean([r["margin"] for r in observations])),
            "queries": len(observations), "per_task": {}
        }
        for tid in range(N_TASKS):
            group = [r for r in observations if r["task_id"] == tid]
            summary[method]["per_task"][str(tid)] = {
                "accuracy": float(np.mean([r["correct"] for r in group])),
                "mean_gold_logprob": float(np.mean([r["gold_logprob"] for r in group]))
            }
    return {"methods": summary,
            "router": {"top1_task_accuracy": route_correct / len(manifest) / 4,
                       "top2_contains_task_rate": route_top2_contains / len(manifest) / 4,
                       "calibration_vectors_per_task": 4,
                       "prototype_count": N_TASKS},
            "native_bias_alias": {"oracle_max_abs_candidate_score_delta": max(alias_differences),
                                  "oracle_score_alias_within_frozen_tolerance": max(alias_differences) <= 0.01},
            "evaluation_queries": len(manifest) * 4, "candidate_sequences": total_candidates,
            "candidate_input_tokens": total_input_tokens, "model_calls": model_calls,
            "inference_seconds_all_methods": inference_seconds}


def _save_npz(path: Path, arrays: dict):
    np.savez(path, **arrays)
    with zipfile.ZipFile(path) as archive:
        assert all(item.compress_type == zipfile.ZIP_STORED for item in archive.infolist())
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def make_payloads(out: Path, model, vectors, prototypes):
    tasks = np.arange(N_TASKS, dtype=np.int16)
    schema = np.array([545, D, ma516.LAYER_OUT, ma516.HOOK_LAYER], dtype=np.int32)
    revision = np.frombuffer(ma516.REVISION.encode(), dtype="S40")
    model_hash = np.frombuffer(ma516.MODEL_SHA.encode(), dtype="S64")
    config_hash = np.frombuffer(ma516.CONFIG_SHA.encode(), dtype="S64")
    tokenizer_hash = np.frombuffer(ma516.TOKENIZER_SHA.encode(), dtype="S64")
    meta = {"task_ids": tasks, "schema": schema, "model_revision": revision,
            "model_sha256": model_hash, "config_sha256": config_hash,
            "tokenizer_sha256": tokenizer_hash, "alpha": np.array([1.0], np.float32)}
    payloads = {}
    def save(name, **arrays):
        size, sha = _save_npz(out / f"{name}.npz", {**meta, **arrays})
        payloads[name] = {"bytes": size, "sha256": sha}
    save("oracle_fv", function_vectors=vectors)
    save("oracle_weight_bias", expert_bias_deltas=vectors)
    save("routed_fv_top1", function_vectors=vectors, router_prototypes=prototypes,
         router=np.array([1, 0], np.float32))
    save("routed_fv_top2", function_vectors=vectors, router_prototypes=prototypes,
         router=np.array([2, 0.05], np.float32))
    save("routed_weight_bias_top1", expert_bias_deltas=vectors, router_prototypes=prototypes,
         router=np.array([1, 0], np.float32))
    save("shared_mean_fv", shared_vector=vectors.mean(axis=0, keepdims=True))
    # Measure a real uncompressed full layer-3 FFN expert bank in a temporary
    # file. It is a bytes-only reference and is not committed as a large artifact.
    mlp_state = model.gpt_neox.layers[ma516.HOOK_LAYER].mlp.state_dict()
    with tempfile.TemporaryDirectory(prefix="ma545-full-mlp-bank-") as td:
        target = Path(td) / "independent_mlp_experts.npz"
        copies = {f"expert_{task}__{name}": value.detach().cpu().numpy()
                  for task in range(N_TASKS) for name, value in mlp_state.items()}
        copies["schema"] = np.array([545, N_TASKS, ma516.HOOK_LAYER], np.int32)
        size, sha = _save_npz(target, copies)
        payloads["full_mlp_expert_bank_bytes_only"] = {"bytes": size, "sha256": sha}
    return payloads


def common_deployment_bytes(model_dir: Path):
    names = ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    details = {name: (model_dir / name).stat().st_size for name in names}
    return sum(details.values()), details


def run(seed: int, model_dir: Path, out_dir: Path):
    if seed not in SEEDS["dev"] + SEEDS["fresh"]:
        raise ValueError("seed is not registered in the frozen protocol")
    if seed in SEEDS["fresh"]:
        gate_path = ROOT / "runs/dev/DEV_GATE.json"
        gate = json.loads(gate_path.read_text())
        if not gate.get("fresh_opened", False):
            raise RuntimeError("fresh seeds are sealed by the preregistered gate")
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tok, torch_module = ma516.load_model(model_dir)
    splits = make_splits(seed)
    t0 = time.perf_counter()
    vectors, prototypes, manifest, extraction = extract_banks(model, tok, torch_module, splits)
    extraction_seconds = time.perf_counter() - t0
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    np.savez(out_dir / "router_features.npz", router_prototypes=prototypes, function_vectors=vectors)
    eval_results = evaluate(model, tok, torch_module, vectors, prototypes, manifest)
    payloads = make_payloads(out_dir, model, vectors, prototypes)
    base_bytes, base_files = common_deployment_bytes(model_dir)
    report = {
        "experiment_id": "MA-545", "seed": seed, "model_revision": ma516.REVISION,
        "model_sha256": ma516.MODEL_SHA, "config_sha256": ma516.CONFIG_SHA,
        "tokenizer_sha256": ma516.TOKENIZER_SHA,
        "support_examples": N_TASKS * 8, "router_calibration_examples": N_TASKS * 4,
        "audit_queries": N_TASKS * 4, "support_extraction": extraction,
        "extraction_seconds": extraction_seconds,
        "common_base_bytes": base_bytes, "common_base_files": base_files,
        "payloads": payloads, "results": eval_results,
        "full_deployment_bytes": {name: (base_bytes + p["bytes"])
                                   for name, p in payloads.items() if name != "full_mlp_expert_bank_bytes_only"},
        "full_mlp_expert_bank_deployment_bytes": base_bytes + payloads["full_mlp_expert_bank_bytes_only"]["bytes"]
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    default_model = Path.home() / ".cache/huggingface/hub/models--EleutherAI--pythia-70m-deduped/snapshots" / ma516.REVISION
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model-dir", type=Path, default=default_model)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = run(args.seed, args.model_dir, args.out)
    print(json.dumps({"seed": args.seed, "results": result["results"], "payloads": result["payloads"]}, indent=2))
