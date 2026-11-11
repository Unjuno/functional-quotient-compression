"""Attention-weighted head matching and shared KV-map coding for MA-878."""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import itertools
import json
import resource
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = ("EleutherAI/pythia-70m-deduped", "HuggingFaceTB/SmolLM2-135M")
PAIR = MODELS
RANKS = (1, 2, 4, 8, 16, 32)
D = 64
ALPHA = 0.01
FIT_MAX_POSITIONS = 64
MIN_TOKENS = 16
MAX_TOKENS = 128
FRESH_SEEDS = (983, 997, 1009)
DEV_SEED = 829
DEV_CAL = 24
HEADER = struct.Struct("<5sBBB")


def snapshot(model_id: str) -> Path:
    root = Path.home() / ".cache/huggingface/hub" / ("models--" + model_id.replace("/", "--")) / "snapshots"
    if not root.exists():
        raise FileNotFoundError(f"cached local model unavailable: {model_id}")
    return sorted(root.iterdir())[0]


def load_assets() -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    models, tokenizers, paths = {}, {}, {}
    for model_id in MODELS:
        path = snapshot(model_id)
        paths[model_id] = path
        tokenizers[model_id] = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
        models[model_id] = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, attn_implementation="eager"
        ).eval()
    return models, tokenizers, paths


def corpus(split: str):
    return load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split, download_mode="reuse_dataset_if_exists")


def eligible_prompts(split: str, tokenizers: dict[str, Any], minimum: int) -> list[dict[str, Any]]:
    ds = corpus(split)
    src_tok, tgt_tok = (tokenizers[x] for x in PAIR)
    prompts, seen = [], set()
    for row_id, row in enumerate(ds):
        text = row["text"].strip()
        if len(text) < 32 or text in seen:
            continue
        a = src_tok(text, add_special_tokens=False, return_offsets_mapping=True)
        b = tgt_tok(text, add_special_tokens=False, return_offsets_mapping=True)
        if a["offset_mapping"] != b["offset_mapping"]:
            continue
        n = len(a["input_ids"])
        if MIN_TOKENS <= n <= MAX_TOKENS:
            seen.add(text)
            prompts.append({"row": row_id, "text": text, "src_ids": a["input_ids"], "tgt_ids": b["input_ids"], "tokens": n})
            if len(prompts) >= minimum:
                break
    return prompts


def select_prompts(split: str, tokenizers: dict[str, Any], seed: int, count: int, role: str) -> list[dict[str, Any]]:
    need = DEV_CAL + 16 if split == "train" else len(FRESH_SEEDS) * count
    prompts = eligible_prompts(split, tokenizers, need)
    pair_tag = sum((i + 1) * ord(c) for i, c in enumerate(PAIR[0] + ">" + PAIR[1]))
    if split == "train":
        permutation = np.random.default_rng(seed + pair_tag % 10000).permutation(len(prompts))
        start = 0 if role == "calibration" else DEV_CAL
        chosen = permutation[start : start + count]
    else:
        salt = 0 if role == "calibration" else 7919
        permutation = np.random.default_rng(51236 + pair_tag + salt).permutation(len(prompts))
        group = FRESH_SEEDS.index(seed)
        chosen = permutation[group * count : (group + 1) * count]
    if len(chosen) < count:
        raise RuntimeError(f"need {count} aligned {split} prompts; found {len(chosen)}")
    return [prompts[int(i)] for i in chosen]


def cache_arrays(model: Any, ids: list[int], output_attentions: bool = False):
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    start = time.perf_counter()
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True, output_attentions=output_attentions)
    elapsed = time.perf_counter() - start
    past = out.past_key_values.to_legacy_cache() if hasattr(out.past_key_values, "to_legacy_cache") else out.past_key_values
    cache = [(k[0].detach().cpu().numpy().astype(np.float64), v[0].detach().cpu().numpy().astype(np.float64)) for k, v in past]
    return cache, out, elapsed


def rope_inv_freq(config: Any, rotary_dim: int) -> np.ndarray:
    theta = float(getattr(config, "rope_theta", 10000.0))
    idx = np.arange(0, rotary_dim, 2, dtype=np.float64)
    return 1.0 / (theta ** (idx / rotary_dim))


def rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def rope_transform(x: np.ndarray, config: Any, inverse: bool = False) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64).copy()
    model_type = getattr(config, "model_type", "")
    rotary_dim = int((config.hidden_size // config.num_attention_heads) * float(config.rotary_pct)) if model_type == "gpt_neox" else y.shape[-1]
    inv = rope_inv_freq(config, rotary_dim)
    pos = np.arange(y.shape[-2], dtype=np.float64)
    ang = np.outer(pos, inv)
    cos = np.concatenate([np.cos(ang), np.cos(ang)], axis=-1)
    sin = np.concatenate([np.sin(ang), np.sin(ang)], axis=-1)
    if model_type == "gpt_neox":
        cos = np.concatenate([cos, np.ones((len(pos), y.shape[-1] - rotary_dim))], axis=-1)
        sin = np.concatenate([sin, np.zeros((len(pos), y.shape[-1] - rotary_dim))], axis=-1)
    shape = (1,) * (y.ndim - 2) + (len(pos), cos.shape[-1])
    cos, sin = cos.reshape(shape), sin.reshape(shape)
    if inverse:
        sin = -sin
    if model_type == "gpt_neox":
        rot, tail = y[..., :rotary_dim], y[..., rotary_dim:]
        return np.concatenate([rot * cos[..., :rotary_dim] + rotate_half(rot) * sin[..., :rotary_dim], tail], axis=-1)
    return y * cos + rotate_half(y) * sin


def target_key_importance(attention: np.ndarray, kv_head: int, q_per_kv: int) -> np.ndarray:
    """Sum causal attention received by each key position for one KV-head group."""
    start = kv_head * q_per_kv
    received = np.asarray(attention[start : start + q_per_kv], dtype=np.float64).sum(axis=(0, 1))
    received = np.maximum(received, 1e-12)
    return received / received.mean()


def source_layer_for_target(tl: int, source_layers: int, target_layers: int) -> int:
    return round(tl * (source_layers - 1) / max(1, target_layers - 1))


def new_stat() -> list[np.ndarray]:
    return [np.zeros((D + 1, D + 1)), np.zeros((D + 1, D)), np.zeros(D), np.zeros(())]


def add_stat(stat: list[np.ndarray], x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> None:
    xa = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    stat[0][:] += xa.T @ (weight[:, None] * xa)
    stat[1][:] += xa.T @ (weight[:, None] * y)
    stat[2][:] += (weight[:, None] * y * y).sum(axis=0)
    stat[3][...] += weight.sum()


def solve_map(stat: list[np.ndarray]) -> np.ndarray:
    eye = np.eye(D + 1)
    eye[-1, -1] = 0.0
    return np.linalg.solve(stat[0] + ALPHA * eye, stat[1])


def relative_fit_error(stat: list[np.ndarray]) -> float:
    weight = stat[3].item()
    if weight <= 0:
        return float("inf")
    w = solve_map(stat)
    rss = stat[2].sum() - 2.0 * np.sum(w * stat[1]) + np.sum(w * (stat[0] @ w))
    return float(max(rss, 0.0) / max(stat[2].sum(), 1e-30))


def fit_candidate_stats(prompts: list[dict[str, Any]], models: dict[str, Any]) -> dict[tuple[int, int, int, int, int], list[np.ndarray]]:
    src, tgt = PAIR
    model_s, model_t = models[src], models[tgt]
    slayers = model_s.config.num_hidden_layers
    tlayers = model_t.config.num_hidden_layers
    sheads = getattr(model_s.config, "num_key_value_heads", None) or model_s.config.num_attention_heads
    theads = getattr(model_t.config, "num_key_value_heads", None) or model_t.config.num_attention_heads
    q_per_kv = model_t.config.num_attention_heads // theads
    stats = {(tl, th, sh, kv): new_stat()
             for tl in range(tlayers) for th in range(theads) for sh in range(sheads) for kv in (0, 1)}
    source_seconds = target_seconds = 0.0
    sample_tokens = 0
    for prompt in prompts:
        source_cache, _, st = cache_arrays(model_s, prompt["src_ids"])
        target_cache, target_out, tt = cache_arrays(model_t, prompt["tgt_ids"], output_attentions=True)
        source_seconds += st
        target_seconds += tt
        n = min(source_cache[0][0].shape[1], target_cache[0][0].shape[1], FIT_MAX_POSITIONS)
        sample_tokens += n
        for tl in range(tlayers):
            sl = source_layer_for_target(tl, slayers, tlayers)
            attention = target_out.attentions[tl][0, :, :n, :n].detach().cpu().numpy()
            for th in range(theads):
                weight = target_key_importance(attention, th, q_per_kv)[:n]
                target_key = target_cache[tl][0][th, :n]
                target_value = target_cache[tl][1][th, :n]
                target_key = rope_transform(target_key, model_t.config, inverse=True)
                for sh in range(sheads):
                    source_key = rope_transform(source_cache[sl][0][sh, :n], model_s.config, inverse=True)
                    source_value = source_cache[sl][1][sh, :n]
                    add_stat(stats[(tl, th, sh, 0)], source_key, target_key, weight)
                    add_stat(stats[(tl, th, sh, 1)], source_value, target_value, weight)
        del source_cache, target_cache, target_out
    stats["_timing"] = {"source_forward_seconds": source_seconds, "target_forward_seconds": target_seconds, "sample_tokens": sample_tokens}
    return stats


def choose_matched_heads(stats: dict[Any, Any], target_layers: int, target_heads: int, source_heads: int) -> list[list[int]]:
    """Per-layer injective assignment minimizing attention-weighted K/V map residual."""
    choices = list(itertools.permutations(range(source_heads), target_heads))
    candidate_error = {}
    for tl in range(target_layers):
        for th in range(target_heads):
            for sh in range(source_heads):
                candidate_error[(tl, th, sh)] = sum(relative_fit_error(stats[(tl, th, sh, kv)]) for kv in (0, 1))
    matched = []
    for tl in range(target_layers):
        best_score, best = float("inf"), None
        for assignment in choices:
            score = sum(candidate_error[(tl, th, sh)] for th, sh in enumerate(assignment))
            if score < best_score:
                best_score, best = score, assignment
        assert best is not None
        matched.append(list(best))
    return matched


def materialize_mapset(stats: dict[Any, Any], assignment: list[list[int]], pair: tuple[str, str], models: dict[str, Any], name: str) -> dict[str, Any]:
    src, tgt = pair
    slayers, tlayers = models[src].config.num_hidden_layers, models[tgt].config.num_hidden_layers
    theads = getattr(models[tgt].config, "num_key_value_heads", None) or models[tgt].config.num_attention_heads
    maps, refs = [], []
    for tl in range(tlayers):
        sl = source_layer_for_target(tl, slayers, tlayers)
        for th in range(theads):
            sh = assignment[tl][th]
            for kv in (0, 1):
                maps.append(solve_map(stats[(tl, th, sh, kv)]))
                refs.append((tl, th, kv, sl, sh))
    return {"pair": pair, "maps": np.stack(maps), "refs": refs, "assignment": assignment, "name": name}


def fit_mapsets(prompts: list[dict[str, Any]], models: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    stats = fit_candidate_stats(prompts, models)
    src, tgt = PAIR
    sheads = getattr(models[src].config, "num_key_value_heads", None) or models[src].config.num_attention_heads
    theads = getattr(models[tgt].config, "num_key_value_heads", None) or models[tgt].config.num_attention_heads
    tlayers = models[tgt].config.num_hidden_layers
    matched_assignment = choose_matched_heads(stats, tlayers, theads, sheads)
    fixed_row = [round(th * (sheads - 1) / max(1, theads - 1)) for th in range(theads)]
    fixed_assignment = [fixed_row[:] for _ in range(tlayers)]
    matched = materialize_mapset(stats, matched_assignment, PAIR, models, "attention_matched")
    fixed = materialize_mapset(stats, fixed_assignment, PAIR, models, "fixed_spread")
    return fixed, matched, stats["_timing"]


def fit_basis(mapset: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    values = mapset["maps"].reshape(len(mapset["maps"]), -1)
    mean = values.mean(axis=0)
    _, _, basis = np.linalg.svd(values - mean, full_matrices=False)
    return mean, basis


def compress_maps(maps: np.ndarray, mean: np.ndarray, basis: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    flat = maps.reshape(len(maps), -1)
    codes = (flat - mean) @ basis[:rank].T
    reconstructed = mean + codes @ basis[:rank]
    return reconstructed.reshape(maps.shape), codes


def apply_maps(source_cache: list[tuple[np.ndarray, np.ndarray]], refs: list[tuple[int, int, int, int, int]], maps: np.ndarray, src_config: Any, tgt_config: Any):
    target_layers = tgt_config.num_hidden_layers
    target_heads = getattr(tgt_config, "num_key_value_heads", None) or tgt_config.num_attention_heads
    length = source_cache[0][0].shape[1]
    out_k = np.zeros((target_layers, target_heads, length, D), dtype=np.float32)
    out_v = np.zeros_like(out_k)
    for i, (tl, th, kv, sl, sh) in enumerate(refs):
        x = source_cache[sl][kv][sh]
        if kv == 0:
            x = rope_transform(x, src_config, inverse=True)
        x_aug = np.concatenate([x, np.ones((length, 1))], axis=1)
        y = (x_aug @ maps[i]).astype(np.float32)
        if kv == 0:
            y = rope_transform(y, tgt_config).astype(np.float32)
        (out_k if kv == 0 else out_v)[tl, th] = y
    return [(torch.from_numpy(out_k[l:l+1]), torch.from_numpy(out_v[l:l+1])) for l in range(target_layers)]


def nll(model: Any, logits: torch.Tensor, label: int) -> float:
    return float(F.cross_entropy(logits.reshape(1, -1), torch.tensor([label], dtype=torch.long)))


def transfer_nll(model: Any, past: list[tuple[torch.Tensor, torch.Tensor]], query: int, label: int) -> tuple[float, float]:
    start = time.perf_counter()
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([[query]], dtype=torch.long), past_key_values=tuple(past), use_cache=True)
    return nll(model, out.logits[0, -1], label), time.perf_counter() - start


def eval_prompts(prompts: list[dict[str, Any]], models: dict[str, Any], fixed: dict[str, Any], matched: dict[str, Any], mean: np.ndarray, basis: np.ndarray, rank: int) -> dict[str, Any]:
    model_s, model_t = (models[x] for x in PAIR)
    compressed, codes = compress_maps(matched["maps"], mean, basis, rank)
    native, fixed_nll, matched_nll, compressed_nll = [], [], [], []
    fixed_ms, matched_ms, compressed_ms, prompt_rows = [], [], [], []
    cache_mse_fixed, cache_mse_matched, cache_mse_compressed = [], [], []
    source_prefill, target_prefill = [], []
    for p in prompts:
        source_cache, _, source_time = cache_arrays(model_s, p["src_ids"])
        target_cache, target_out, target_time = cache_arrays(model_t, p["tgt_ids"])
        source_prefill.append(source_time); target_prefill.append(target_time)
        native.append(nll(model_t, target_out.logits[0, -2], p["tgt_ids"][-1]))
        past_fixed = apply_maps(source_cache, fixed["refs"], fixed["maps"], model_s.config, model_t.config)
        past_matched = apply_maps(source_cache, matched["refs"], matched["maps"], model_s.config, model_t.config)
        t0 = time.perf_counter(); past_compressed = apply_maps(source_cache, matched["refs"], compressed, model_s.config, model_t.config); compressed_ms.append(time.perf_counter()-t0)
        t0 = time.perf_counter(); _ = apply_maps(source_cache, fixed["refs"], fixed["maps"], model_s.config, model_t.config); fixed_ms.append(time.perf_counter()-t0)
        t0 = time.perf_counter(); _ = apply_maps(source_cache, matched["refs"], matched["maps"], model_s.config, model_t.config); matched_ms.append(time.perf_counter()-t0)
        prefix_len = len(p["tgt_ids"]) - 2
        trim = lambda past: [(k[:, :, :prefix_len].contiguous(), v[:, :, :prefix_len].contiguous()) for k, v in past]
        query, label = p["tgt_ids"][-2], p["tgt_ids"][-1]
        fixed_nll.append(transfer_nll(model_t, trim(past_fixed), query, label)[0])
        matched_nll.append(transfer_nll(model_t, trim(past_matched), query, label)[0])
        compressed_nll.append(transfer_nll(model_t, trim(past_compressed), query, label)[0])
        for past, dest in ((past_fixed, cache_mse_fixed), (past_matched, cache_mse_matched), (past_compressed, cache_mse_compressed)):
            err = den = 0.0
            for layer, (pk, pv) in enumerate(past):
                tk, tv = target_cache[layer]
                err += float(np.sum((pk[0].numpy()-tk)**2) + np.sum((pv[0].numpy()-tv)**2))
                den += float(np.sum(tk**2) + np.sum(tv**2))
            dest.append(err / max(den, 1e-30))
        prompt_rows.append(p["row"])
        del source_cache, target_cache, target_out
    return {
        "native_nll": native, "fixed_spread_nll": fixed_nll, "matched_full_nll": matched_nll, "matched_rank_nll": compressed_nll,
        "fixed_delta": float(np.mean(fixed_nll)-np.mean(native)), "matched_full_delta": float(np.mean(matched_nll)-np.mean(native)),
        "matched_rank_delta": float(np.mean(compressed_nll)-np.mean(native)),
        "matched_gain_vs_fixed": float(np.mean(fixed_nll)-np.mean(matched_nll)),
        "rank_gap_vs_full": float(np.mean(compressed_nll)-np.mean(matched_nll)),
        "cache_mse_fixed": float(np.mean(cache_mse_fixed)), "cache_mse_matched": float(np.mean(cache_mse_matched)),
        "cache_mse_rank": float(np.mean(cache_mse_compressed)), "fixed_apply_seconds": fixed_ms,
        "matched_apply_seconds": matched_ms, "rank_apply_seconds": compressed_ms, "prompt_rows": prompt_rows, "map_codes": codes.tolist(),
        "source_prefill_seconds": source_prefill, "native_target_prefill_seconds": target_prefill,
    }


def serialize_payloads(fixed: dict[str, Any], matched: dict[str, Any], mean: np.ndarray, basis: np.ndarray, rank: int) -> dict[str, int]:
    pair_name = (PAIR[0] + ">" + PAIR[1]).encode() + b"\0"
    fixed_meta = bytes(v for r in fixed["refs"] for v in r)
    match_meta = bytes(v for r in matched["refs"] for v in r)
    full_fixed = HEADER.pack(b"MA878", 0, 0, 1) + pair_name + np.asarray(fixed["maps"], dtype="<f4").tobytes() + fixed_meta
    full_matched = HEADER.pack(b"MA878", 1, 0, 1) + pair_name + np.asarray(matched["maps"], dtype="<f4").tobytes() + match_meta
    _, codes = compress_maps(matched["maps"], mean, basis, rank)
    shared = HEADER.pack(b"MA878", 2, rank, 1) + pair_name + np.asarray(mean, dtype="<f4").tobytes()
    shared += np.asarray(basis[:rank], dtype="<f4").tobytes() + np.asarray(codes, dtype="<f4").tobytes() + match_meta
    return {"fixed_spread_full": len(full_fixed), "attention_matched_full": len(full_matched), "attention_matched_rank": len(shared)}


def file_provenance(paths: dict[str, Path]) -> dict[str, Any]:
    out = {}
    for name, path in paths.items():
        weight = path / "model.safetensors"
        out[name] = {"snapshot": path.name, "weight_bytes": weight.stat().st_size if weight.exists() else 0}
    return out


def run_world(seed: int, cal_split: str, test_split: str, ncal: int, ntest: int, ranks: tuple[int, ...], models: dict[str, Any], tokenizers: dict[str, Any]) -> dict[str, Any]:
    calibration_start = time.perf_counter()
    calibration = select_prompts(cal_split, tokenizers, seed, ncal, "calibration")
    fixed, matched, timing = fit_mapsets(calibration, models)
    timing["total_calibration_seconds"] = time.perf_counter() - calibration_start
    basis_start = time.perf_counter()
    mean, basis = fit_basis(matched)
    basis_seconds = time.perf_counter() - basis_start
    test = select_prompts(test_split, tokenizers, seed, ntest, "test")
    rank_results = {str(r): eval_prompts(test, models, fixed, matched, mean, basis, r) for r in ranks}
    payloads = {str(r): serialize_payloads(fixed, matched, mean, basis, r) for r in ranks}
    assignment = matched["assignment"]
    return {
        "seed": seed,
        "calibration_rows": [p["row"] for p in calibration],
        "test_rows": [p["row"] for p in test],
        "calibration_tokens": sum(p["tokens"] for p in calibration),
        "test_tokens": sum(p["tokens"] for p in test),
        "calibration_timing": timing,
        "basis_fit_seconds": basis_seconds,
        "rank_results": rank_results,
        "payload_bytes_by_rank": payloads,
        "matched_head_assignments": assignment,
        "fixed_head_assignment": fixed["assignment"],
        "map_count": len(matched["maps"]),
    }


def run_dev() -> dict[str, Any]:
    models, tokenizers, paths = load_assets()
    row = run_world(DEV_SEED, "train", "train", 24, 16, RANKS, models, tokenizers)
    return {"experiment_id": "MA-878", "split": "Wikitext-2 train", "rank_candidates": list(RANKS), "row": row,
            "models": file_provenance(paths), "runtime": {"cuda_available": torch.cuda.is_available(), "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}}


def run_fresh(seeds: tuple[int, ...], rank: int) -> dict[str, Any]:
    models, tokenizers, paths = load_assets()
    rows = [run_world(seed, "validation", "test", 12, 16, (rank,), models, tokenizers) for seed in seeds]
    return {"experiment_id": "MA-878", "seeds": list(seeds), "rank": rank, "rows": rows,
            "models": file_provenance(paths), "runtime": {"cuda_available": torch.cuda.is_available(), "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}}


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
