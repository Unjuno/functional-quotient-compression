"""Frozen MA-464 development screen with audit bytes unopened until gate pass."""
from __future__ import annotations
import csv, hashlib, json, math, random, statistics, time
from pathlib import Path

import torch
from torch.nn import functional as F

from acquire_corpora import DATA, acquire_all
from model import AdaptedGPT, MODES, make_gpt_config, _nano

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
ARTIFACTS = SOURCE / "artifacts"
DATA_DIR = SOURCE / "data"
ARTIFACTS.mkdir(exist_ok=True)
SEEDS = (46401, 46402)
BLOCK, BATCH = 128, 16
BASE_UPDATES, ADAPTER_UPDATES = 1200, 600
BASE_LR, ADAPTER_LR, BASE_WD = 0.0006, 0.003, 0.1
RANK, LAYERS, WIDTH = 4, 4, 64
EVAL_BATCHES = 8
DOMAINS = ("shakespeare", "austen")
SOURCE_URLS = {
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "austen": "https://www.gutenberg.org/files/1342/1342-0.txt",
}


def read_prefix_splits(manifests):
    raw, char_text, split_sizes = {}, {}, {}
    for name in DOMAINS:
        path = DATA_DIR / f"{name}.txt"
        size = path.stat().st_size
        train_end, dev_end = int(size * .8), int(size * .9)
        with path.open("rb") as f:
            train_raw = f.read(train_end)
            dev_raw = f.read(dev_end - train_end)
        train_text = train_raw.decode("utf-8", errors="ignore")
        dev_text = dev_raw.decode("utf-8", errors="ignore")
        char_text[name] = (train_text, dev_text)
        split_sizes[name] = {"train_bytes": len(train_raw), "dev_bytes": len(dev_raw),
                             "train_chars": len(train_text), "dev_chars": len(dev_text),
                             "audit_bytes_not_read": size - dev_end}
    vocab = sorted(set("".join(char_text[n][0] for n in DOMAINS)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    unk = len(vocab)
    tokens = {}
    for name in DOMAINS:
        train_text, dev_text = char_text[name]
        tokens[name] = {
            "train": torch.tensor([stoi.get(ch, unk) for ch in train_text], dtype=torch.long),
            "dev": torch.tensor([stoi.get(ch, unk) for ch in dev_text], dtype=torch.long),
        }
    vocab_with_unk = vocab + ["<UNK>"]
    return tokens, vocab_with_unk, split_sizes


def open_audit_tokens(vocab):
    stoi = {ch: i for i, ch in enumerate(vocab[:-1])}
    unk = len(vocab) - 1
    audit, meta = {}, {}
    for name in DOMAINS:
        path = DATA_DIR / f"{name}.txt"
        size = path.stat().st_size
        dev_end = int(size * .9)
        with path.open("rb") as f:
            f.seek(dev_end)
            raw = f.read()
        text = raw.decode("utf-8", errors="ignore")
        audit[name] = torch.tensor([stoi.get(ch, unk) for ch in text], dtype=torch.long)
        meta[name] = {"audit_bytes": len(raw), "audit_characters": len(text), "audit_start_byte": dev_end}
    return audit, meta


def sample_starts(length, count, generator):
    if length <= BLOCK + 1:
        raise ValueError("token split is too short")
    return torch.randint(0, length - BLOCK - 1, (count,), generator=generator)


def batch_at(tokens, starts):
    offs = torch.arange(BLOCK + 1)
    seq = tokens[starts[:, None] + offs[None, :]]
    return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()


def eval_lm(model, tokens, seed, view_id=None):
    if isinstance(model, AdaptedGPT):
        model.set_view(int(view_id or 0), count=False)
    model.eval()
    starts = sample_starts(tokens.numel(), BATCH * EVAL_BATCHES,
                           torch.Generator(device="cpu").manual_seed(seed))
    total, count = 0.0, 0
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(starts), BATCH):
            x, y = batch_at(tokens, starts[i:i+BATCH])
            logits, _ = model(x, y)
            total += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum"))
            count += y.numel()
    elapsed = time.perf_counter() - start
    model.train()
    return {"nll": total / count, "bits_per_character": total / count / math.log(2),
            "evaluation_wall_seconds": elapsed, "tokens_per_second": count / max(elapsed, 1e-12)}


def make_batches(tokens, count, generator):
    starts = sample_starts(tokens.numel(), count, generator)
    return batch_at(tokens, starts)


def base_training(seed, tokens, vocab):
    torch.manual_seed(seed)
    config = make_gpt_config(len(vocab), BLOCK)
    model = _nano.GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=BASE_WD)
    gen = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    model.train(); start = time.perf_counter()
    for _ in range(BASE_UPDATES):
        x, y = make_batches(tokens["shakespeare"]["train"], BATCH, gen)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    elapsed = time.perf_counter() - start
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    return config, state, elapsed


def save_base_payload(seed, config, state, vocab, manifests):
    path = ARTIFACTS / f"seed{seed}_base_inference.pt"
    payload = {"schema": "MA-464/base-inference-v1", "seed": seed,
               "config": config.__dict__, "vocabulary": vocab, "corpora": manifests,
               "model_state": state}
    torch.save(payload, path)
    raw = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def adapt_one(seed, condition, config, base_state, tokens, vocab, manifests):
    torch.manual_seed(seed + 20_000 + MODES.index(condition))
    model = AdaptedGPT(base_state, config, condition, len(vocab), rank=RANK)
    trainable = model.adapter_parameters()
    if not trainable or any(not p.requires_grad for p in trainable):
        raise AssertionError("adapter parameters were not enabled")
    optimizer = torch.optim.AdamW(trainable, lr=ADAPTER_LR, weight_decay=0.0)
    data_gen = torch.Generator(device="cpu").manual_seed(seed + 30_000)
    view_gen = torch.Generator(device="cpu").manual_seed(seed + 40_000 + MODES.index(condition))
    domain_counts = {n: 0 for n in DOMAINS}
    per_view_counts = [0, 0]
    model.train(); start = time.perf_counter()
    for _ in range(ADAPTER_UPDATES):
        domain_index = int(torch.randint(0, 2, (1,), generator=data_gen))
        domain = DOMAINS[domain_index]
        domain_counts[domain] += 1
        if condition == "single":
            view_id = 0
        else:
            view_id = int(torch.randint(0, 2, (1,), generator=view_gen))
            per_view_counts[view_id] += 1
        model.set_view(view_id)
        x, y = make_batches(tokens[domain]["train"], BATCH, data_gen)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step()
    train_wall = time.perf_counter() - start
    # Evaluate every logical function on the same domain-specific dev windows.
    n_views = 1 if condition == "single" else 2
    per_view = {}
    for view_id in range(n_views):
        per_view[str(view_id)] = {}
        for j, domain in enumerate(DOMAINS):
            per_view[str(view_id)][domain] = eval_lm(model, tokens[domain]["dev"],
                seed + 50_000 + 100 * j, view_id=view_id)
    view_logit_distance = compare_view_logits(model, tokens["shakespeare"]["dev"], seed + 60_000) if n_views == 2 else 0.0
    merged = model.merged_model(tuple(range(n_views)))
    merged_metrics = {domain: eval_lm(merged, tokens[domain]["dev"], seed + 50_000 + 100 * j)
                      for j, domain in enumerate(DOMAINS)}
    bank_payload = save_bank_payload(seed, condition, model, vocab, manifests, domain_counts, per_view_counts)
    merged_payload = save_merged_payload(seed, condition, merged, vocab, manifests)
    adapter_payload = save_adapter_payload(seed, condition, model)
    all_view_values = [v[domain]["nll"] for v in per_view.values() for domain in DOMAINS]
    merged_mean = statistics.mean(merged_metrics[d]["nll"] for d in DOMAINS)
    active_macs = LAYERS * (2 * WIDTH * RANK + 2 * RANK * 3 * WIDTH)
    return {"seed": seed, "condition": condition, "updates": ADAPTER_UPDATES,
            "training_tokens_seen": ADAPTER_UPDATES * BATCH * BLOCK,
            "domain_update_counts": domain_counts,
            "route_update_counts": per_view_counts,
            "train_wall_seconds": train_wall,
            "active_lora_macs_per_token_proxy": active_macs,
            "per_view": per_view, "mean_per_view_nll": statistics.mean(all_view_values),
            "merged": merged_metrics, "merged_mean_nll": merged_mean,
            "view_logit_distance": view_logit_distance,
            "bank_payload": bank_payload, "adapter_payload": adapter_payload,
            "merged_payload": merged_payload}


def compare_view_logits(model, tokens, seed):
    starts = sample_starts(tokens.numel(), BATCH, torch.Generator(device="cpu").manual_seed(seed))
    x, y = batch_at(tokens, starts)
    model.eval(); outputs = []
    with torch.no_grad():
        for view in (0, 1):
            model.set_view(view, count=False)
            logits, _ = model(x, y)
            outputs.append(logits)
    model.train()
    return float((outputs[0] - outputs[1]).abs().mean())


def state_cpu(model):
    return {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}


def write_payload(path, payload):
    torch.save(payload, path)
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def inference_config(config, condition, mode):
    return {"layers": config.n_layer, "width": config.n_embd, "heads": config.n_head,
            "block_size": config.block_size, "vocab_size": config.vocab_size,
            "adapter_target": "attn.c_attn QKV", "rank": RANK,
            "condition": condition, "mode": mode}


def save_bank_payload(seed, condition, model, vocab, manifests, domain_counts, route_counts):
    path = ARTIFACTS / f"seed{seed}_{condition}_two_view_bank.pt"
    payload = {"schema": "MA-464/two-view-inference-bank-v1", "seed": seed,
               "config": inference_config(model.config, condition, "two_view_bank"),
               "vocabulary": vocab, "corpora": manifests,
               "route_metadata": {"domain_updates": domain_counts, "view_updates": route_counts,
                                  "view_ids": [0, 1] if condition != "single" else [0]},
               "model_state": state_cpu(model)}
    full = write_payload(path, payload)
    adapter_path = ARTIFACTS / f"seed{seed}_{condition}_adapter_bank.pt"
    adapter_info = write_payload(adapter_path, {"schema": "MA-464/adapter-bank-v1", "condition": condition,
                                                 "adapter_state": model.bank_adapter_state()})
    full["adapter_only_bytes"] = adapter_info["bytes"]
    full["adapter_only_sha256"] = adapter_info["sha256"]
    return full


def save_adapter_payload(seed, condition, model):
    # Detail is nested in the bank payload metadata; this records exact additive bytes.
    info_path = ARTIFACTS / f"seed{seed}_{condition}_adapter_bank.pt"
    return {"bytes": info_path.stat().st_size, "sha256": hashlib.sha256(info_path.read_bytes()).hexdigest()}


def save_merged_payload(seed, condition, model, vocab, manifests):
    path = ARTIFACTS / f"seed{seed}_{condition}_merged_inference.pt"
    payload = {"schema": "MA-464/merged-inference-v1", "seed": seed,
               "config": inference_config(model.config, condition, "merged"),
               "vocabulary": vocab, "corpora": manifests,
               "model_state": state_cpu(model)}
    return write_payload(path, payload)


def base_condition_result(seed, config, state, tokens, vocab, manifests, base_payload, base_wall):
    model = _nano.GPT(config); model.load_state_dict(state)
    dev = {domain: eval_lm(model, tokens[domain]["dev"], seed + 50_000 + 100 * j)
           for j, domain in enumerate(DOMAINS)}
    mean = statistics.mean(dev[d]["nll"] for d in DOMAINS)
    return {"seed": seed, "condition": "base_only", "updates": BASE_UPDATES,
            "training_tokens_seen": BASE_UPDATES * BATCH * BLOCK,
            "domain_update_counts": {"shakespeare": BASE_UPDATES, "austen": 0},
            "route_update_counts": [], "train_wall_seconds": base_wall,
            "active_lora_macs_per_token_proxy": 0, "per_view": {"0": dev},
            "mean_per_view_nll": mean, "merged": dev, "merged_mean_nll": mean,
            "view_logit_distance": 0.0, "bank_payload": base_payload,
            "adapter_payload": {"bytes": 0, "sha256": None}, "merged_payload": base_payload}


def evaluate_audit_if_eligible(gates, seeds, states, configs, vocab, manifests):
    if not gates["audit_eligible"]:
        return None
    audit, audit_meta = open_audit_tokens(vocab)
    results = []
    # Models are reloaded from immutable inference payloads; no tuning follows.
    for seed in seeds:
        for condition in ("adamix", "mirror", "film"):
            bank_file = ARTIFACTS / f"seed{seed}_{condition}_two_view_bank.pt"
            bank = torch.load(bank_file, map_location="cpu", weights_only=False)
            model = AdaptedGPT(states[seed], configs[seed], condition, len(vocab), rank=RANK)
            model.load_state_dict(bank["model_state"])
            per_view = {}
            for view in (0, 1):
                per_domain = {domain: eval_lm(model, audit[domain], seed + 70_000 + 100*j, view)
                              for j, domain in enumerate(DOMAINS)}
                per_view[str(view)] = per_domain
            merged_file = ARTIFACTS / f"seed{seed}_{condition}_merged_inference.pt"
            merged_payload = torch.load(merged_file, map_location="cpu", weights_only=False)
            merged_model = _nano.GPT(configs[seed])
            merged_model.load_state_dict(merged_payload["model_state"])
            merged_metrics = {domain: eval_lm(merged_model, audit[domain], seed + 70_000 + 100*j)
                              for j, domain in enumerate(DOMAINS)}
            per_view_mean = statistics.mean(m[domain]["nll"] for m in per_view.values() for domain in DOMAINS)
            merged_mean = statistics.mean(merged_metrics[d]["nll"] for d in DOMAINS)
            results.append({"seed": seed, "condition": condition, "per_view": per_view,
                            "mean_per_view_nll": per_view_mean, "merged": merged_metrics,
                            "merged_mean_nll": merged_mean})
    by = {(r["seed"], r["condition"]): r for r in results}
    checks = []
    for seed in seeds:
        mirror, ada, film = by[(seed, "mirror")], by[(seed, "adamix")], by[(seed, "film")]
        checks.append({"seed": seed,
            "per_view_quality_vs_adamix": mirror["mean_per_view_nll"] <= ada["mean_per_view_nll"] + .10,
            "merged_quality_vs_adamix": mirror["merged_mean_nll"] <= ada["merged_mean_nll"] + .10,
            "per_view_beats_film": mirror["mean_per_view_nll"] <= film["mean_per_view_nll"] - .02,
            "merged_beats_film": mirror["merged_mean_nll"] <= film["merged_mean_nll"] - .02})
    return {"audit_opened": True, "audit_split": audit_meta, "checks": checks,
            "audit_gate_pass": all(all(v for k, v in c.items() if k != "seed") for c in checks),
            "results": results}


def calculate_gates(rows):
    by = {(r["seed"], r["condition"]): r for r in rows}
    checks = []
    for seed in SEEDS:
        mirror, ada, film = by[(seed, "mirror")], by[(seed, "adamix")], by[(seed, "film")]
        bank_match = abs(mirror["bank_payload"]["bytes"] - film["bank_payload"]["bytes"]) <= .05 * mirror["bank_payload"]["bytes"]
        checks.append({"seed": seed,
            "per_view_quality_vs_adamix": mirror["mean_per_view_nll"] <= ada["mean_per_view_nll"] + .10,
            "merged_quality_vs_adamix": mirror["merged_mean_nll"] <= ada["merged_mean_nll"] + .10,
            "complete_bank_bytes": mirror["bank_payload"]["bytes"] <= .99 * ada["bank_payload"]["bytes"],
            "adapter_only_bytes": mirror["adapter_payload"]["bytes"] <= .55 * ada["adapter_payload"]["bytes"],
            "fiLM_byte_match": bank_match,
            "per_view_beats_film": bank_match and mirror["mean_per_view_nll"] <= film["mean_per_view_nll"] - .02,
            "merged_beats_film": bank_match and mirror["merged_mean_nll"] <= film["merged_mean_nll"] - .02})
    passed = all(all(v for k, v in check.items() if k != "seed") for check in checks)
    return {"development_gate_pass": passed, "audit_eligible": passed, "checks": checks}


def main():
    torch.set_num_threads(1)
    manifests = acquire_all()
    tokens, vocab, split_sizes = read_prefix_splits(manifests)
    all_rows, base_states, configs = [], {}, {}
    for seed in SEEDS:
        config, state, base_wall = base_training(seed, tokens, vocab)
        configs[seed], base_states[seed] = config, state
        base_payload = save_base_payload(seed, config, state, vocab, manifests)
        base_row = base_condition_result(seed, config, state, tokens, vocab, manifests, base_payload, base_wall)
        all_rows.append(base_row)
        print(f"DONE seed={seed} base_only mean_nll={base_row['mean_per_view_nll']:.6f} wall={base_wall:.1f}s", flush=True)
        for condition in MODES:
            print(f"START seed={seed} condition={condition}", flush=True)
            row = adapt_one(seed, condition, config, state, tokens, vocab, manifests)
            all_rows.append(row)
            print(f"DONE seed={seed} condition={condition} view_nll={row['mean_per_view_nll']:.6f} merged_nll={row['merged_mean_nll']:.6f} bank_bytes={row['bank_payload']['bytes']} adapter_bytes={row['adapter_payload']['bytes']} wall={row['train_wall_seconds']:.1f}s", flush=True)
    gates = calculate_gates(all_rows)
    audit = evaluate_audit_if_eligible(gates, SEEDS, base_states, configs, vocab, manifests)
    summary = {"corpora": manifests, "split_sizes": split_sizes, "vocabulary_size": len(vocab),
               "audit_opened": audit is not None, "results": all_rows, "gates": gates, "audit": audit,
               "protocol": {"base_updates": BASE_UPDATES, "adaptation_updates": ADAPTER_UPDATES,
                            "seeds": SEEDS, "domains": DOMAINS}}
    (SOURCE / "development_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_core_csv(all_rows, audit)
    print(json.dumps(gates, indent=2), flush=True)


def write_core_csv(rows, audit=None):
    path = ROOT / "RESULTS_CORE.csv"
    fields = ["condition", "world_or_seed", "method", "serialized_bytes", "adapter_only_bytes",
              "train_tokens_or_examples", "optimizer_updates", "active_compute_proxy", "wall_time_s",
              "primary_metric", "primary_value", "secondary_metric", "secondary_value", "status_note"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for row in rows:
            writer.writerow({"condition": row["condition"], "world_or_seed": row["seed"],
                "method": "MA-464 frozen development", "serialized_bytes": row["bank_payload"]["bytes"],
                "adapter_only_bytes": row["adapter_payload"]["bytes"],
                "train_tokens_or_examples": row["training_tokens_seen"], "optimizer_updates": row["updates"],
                "active_compute_proxy": row["active_lora_macs_per_token_proxy"], "wall_time_s": row["train_wall_seconds"],
                "primary_metric": "mean per-view dev NLL", "primary_value": row["mean_per_view_nll"],
                "secondary_metric": "merged mean dev NLL", "secondary_value": row["merged_mean_nll"],
                "status_note": "audit unopened unless all dev gates pass"})
        if audit is not None:
            by = {(r["seed"], r["condition"]): r for r in rows}
            for audited in audit["results"]:
                source = by[(audited["seed"], audited["condition"])]
                wall = sum(m[d]["evaluation_wall_seconds"] for m in audited["per_view"].values() for d in DOMAINS)
                wall += sum(audited["merged"][d]["evaluation_wall_seconds"] for d in DOMAINS)
                writer.writerow({"condition": audited["condition"], "world_or_seed": f"audit-{audited['seed']}",
                    "method": "locked audit", "serialized_bytes": source["bank_payload"]["bytes"],
                    "adapter_only_bytes": source["adapter_payload"]["bytes"], "train_tokens_or_examples": 0,
                    "optimizer_updates": 0, "active_compute_proxy": source["active_lora_macs_per_token_proxy"],
                    "wall_time_s": wall, "primary_metric": "audit mean per-view NLL",
                    "primary_value": audited["mean_per_view_nll"], "secondary_metric": "audit merged mean NLL",
                    "secondary_value": audited["merged_mean_nll"], "status_note": "locked audit; no tuning"})


if __name__ == "__main__":
    main()
