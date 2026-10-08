"""Frozen fixed-budget development screen for MA-784.

The audit span is never tokenized by this script. It is left sealed unless the
development gates pass in both seeds.
"""
from __future__ import annotations

import csv, hashlib, json, math, statistics, time
from pathlib import Path

import torch
from torch.nn import functional as F

from acquire_corpus import DATA, acquire
from model import CONDITIONS, SHAPES, ModelConfig, SmallGPT, estimate_normrouter_c

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
ARTIFACTS = SOURCE / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
SEEDS = (78401, 78402)
WIDTH, LAYERS, HEADS, LOGICAL, FF = 64, 4, 4, 4, 128
BLOCK, BATCH, UPDATES, EVAL_EVERY, EVAL_BATCHES = 128, 16, 1200, 100, 8
LR, WEIGHT_DECAY, AUX_WEIGHT = 0.0006, 0.1, 0.01
GATE_GROUPS = 8


def make_config():
    return ModelConfig(layers=LAYERS, width=WIDTH, heads=HEADS, experts=LOGICAL,
                       expert_hidden=FF, block_size=BLOCK, dropout=0.0,
                       gate_groups=GATE_GROUPS, pool_aux_weight=AUX_WEIGHT,
                       normrouter_c_by_logical={n: estimate_normrouter_c(n, 1) for n in (2, 4)})


def split_and_encode(manifest):
    path = DATA
    raw_size = path.stat().st_size
    train_end, dev_end = int(raw_size * .8), int(raw_size * .9)
    # Read/decoding is bounded at the audit boundary. The final 10% is not read.
    with path.open("rb") as f:
        train_raw = f.read(train_end)
        f.seek(train_end)
        dev_raw = f.read(dev_end - train_end)
    train_text = train_raw.decode("utf-8", errors="ignore")
    dev_text = dev_raw.decode("utf-8", errors="ignore")
    vocab = sorted(set(train_text))
    stoi = {char: i for i, char in enumerate(vocab)}
    unk = len(vocab)
    train = torch.tensor([stoi.get(ch, unk) for ch in train_text], dtype=torch.long)
    dev = torch.tensor([stoi.get(ch, unk) for ch in dev_text], dtype=torch.long)
    vocab_with_unk = vocab + ["<UNK>"]
    manifest.update({"train_bytes": len(train_raw), "dev_bytes": len(dev_raw),
                     "train_chars": len(train_text), "dev_chars": len(dev_text),
                     "vocabulary_size": len(vocab_with_unk),
                     "audit_bytes_not_read": raw_size - dev_end})
    return train, dev, vocab_with_unk


def open_audit_tokens(vocab):
    """Read/tokenize the sealed audit span; caller must first pass all dev gates."""
    raw_size = DATA.stat().st_size
    train_end, dev_end = int(raw_size * .8), int(raw_size * .9)
    with DATA.open("rb") as f:
        f.seek(dev_end)
        audit_raw = f.read()
    audit_text = audit_raw.decode("utf-8", errors="ignore")
    stoi = {char: i for i, char in enumerate(vocab[:-1])}
    unk = len(vocab) - 1
    tokens = torch.tensor([stoi.get(ch, unk) for ch in audit_text], dtype=torch.long)
    return tokens, {"audit_bytes": len(audit_raw), "audit_characters": len(audit_text),
                    "audit_start_byte": dev_end, "audit_end_byte": raw_size,
                    "audit_bytes_read_after_gate": True}


def sample_positions(length: int, count: int, seed: int):
    if length <= BLOCK + 1:
        raise ValueError(f"split too short for block: {length}")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, length - BLOCK - 1, (count,), generator=gen)


def batch_at(tokens, positions):
    offsets = torch.arange(BLOCK + 1)
    seq = tokens[positions[:, None] + offsets[None, :]]
    return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()


def eval_model(model, tokens, positions):
    model.eval(); loss_sum = 0.0; count = 0; loads = []; entropies = []
    start = time.perf_counter()
    with torch.no_grad():
        for start_pos in range(0, len(positions), BATCH):
            pos = positions[start_pos:start_pos + BATCH]
            x, y = batch_at(tokens, pos)
            logits, _, _, load, entropy = model(x, y)
            loss_sum += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum"))
            count += y.numel()
            if load.numel() > 1:
                loads.append(load); entropies.append(float(entropy))
    elapsed = time.perf_counter() - start
    model.train()
    return {"nll": loss_sum / count, "bits_per_character": loss_sum / count / math.log(2),
            "physical_load": torch.stack(loads).mean(0).tolist() if loads else [],
            "router_entropy": statistics.mean(entropies) if entropies else 0.0,
            "evaluation_wall_seconds": elapsed, "evaluation_tokens_per_second": count / max(elapsed, 1e-12)}


def init_models(seed, vocab_size):
    cfg = make_config()
    torch.manual_seed(seed)
    pool4 = SmallGPT(cfg, "unipool4", vocab_size)
    models = {"unipool4": pool4}
    for condition in CONDITIONS:
        if condition == "unipool4":
            continue
        torch.manual_seed(seed + 100 + CONDITIONS.index(condition))
        model = SmallGPT(cfg, condition, vocab_size)
        src, dst = pool4.state_dict(), model.state_dict()
        for key, value in dst.items():
            if key in src and src[key].shape == value.shape and (key.startswith(("wte.", "wpe.", "ln_f.")) or ".ln_1." in key or ".attn." in key or ".ln_2." in key):
                value.copy_(src[key])
            elif key.startswith("shared_experts."):
                # Smaller physical pools start from the corresponding native UniPool experts.
                dst_id = int(key.split(".")[1])
                src_key = key.replace(f"shared_experts.{dst_id}.", f"shared_experts.{dst_id % 4}.")
                if src_key in src and src[src_key].shape == value.shape:
                    value.copy_(src[src_key])
            elif key.startswith("layers.") and ".local_experts." in key:
                expert_id = int(key.split(".local_experts.", 1)[1].split(".", 1)[0])
                suffix = key.split(".local_experts.", 1)[1].split(".", 1)[1]
                src_key = f"shared_experts.{expert_id}.{suffix}"
                if src_key in src and src[src_key].shape == value.shape:
                    value.copy_(src[src_key])
            elif key.startswith("layers.") and ".dense_mlp." in key:
                suffix = key.split(".dense_mlp.", 1)[1]
                src_key = f"shared_experts.0.{suffix}"
                if src_key in src and src[src_key].shape == value.shape:
                    value.copy_(src[src_key])
            elif key.startswith("layers.") and ".router." in key:
                src_key = key
                if src_key in src:
                    n = min(src[src_key].shape[0], value.shape[0])
                    value[:n].copy_(src[src_key][:n])
        model.load_state_dict(dst)
        models[condition] = model
    return models


def active_macs_per_token(condition):
    physical, logical, mode = SHAPES[condition]
    d, t, ff, l = WIDTH, BLOCK, FF, LAYERS
    base = l * (4 * d * d + 2 * t * d)
    router = l * (d * logical + logical + d * logical)
    expert = l * 2 * d * ff
    view = l * (48 if mode == "mirror" else d if mode == "film" else 0)
    return base + router + expert + view


def serialize(seed, condition, model, vocab, manifest):
    physical, logical, mode = SHAPES[condition]
    config = {"layers": LAYERS, "width": WIDTH, "heads": HEADS, "logical_routes": logical,
              "physical_experts": physical, "expert_hidden": FF, "block_size": BLOCK,
              "condition": condition, "mode": mode, "gate_groups": GATE_GROUPS,
              "normrouter_c": model.config.normrouter_c_by_logical.get(logical), "routing": "top1-PA205-NormRouter" if mode != "dense" else "none"}
    payload = {"schema": "MA-784/inference-v1", "seed": seed, "condition": condition,
               "config": config, "vocabulary": vocab, "corpus_sha256": manifest["sha256"],
               "model_state": {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}}
    path = ARTIFACTS / f"seed{seed}_{condition}_inference.pt"
    torch.save(payload, path)
    raw = path.read_bytes()
    try:
        display_path = str(path.relative_to(ROOT))
    except ValueError:
        display_path = str(path)
    bank_bytes = sum(v.numel() * v.element_size() for k, v in payload["model_state"].items()
                     if k.startswith("shared_experts.") or (".local_experts." in k))
    return {"path": display_path, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "expert_bank_tensor_bytes": bank_bytes}


def train_one(seed, condition, model, train, dev, vocab, corpus_manifest):
    torch.manual_seed(seed + 10000 + CONDITIONS.index(condition))
    train_gen = torch.Generator(device="cpu").manual_seed(seed + 20000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    dev_positions = sample_positions(dev.numel(), BATCH * EVAL_BATCHES, seed + 30000)
    best_nll, best_state, history = float("inf"), None, []
    start = time.perf_counter(); model.train()
    for step in range(1, UPDATES + 1):
        positions = sample_positions(train.numel(), BATCH, int(torch.randint(0, 2**31 - 1, (1,), generator=train_gen)))
        x, y = batch_at(train, positions)
        _, lm_loss, aux, _, _ = model(x, y)
        loss = lm_loss + AUX_WEIGHT * aux
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step % EVAL_EVERY == 0:
            row = eval_model(model, dev, dev_positions); row["step"] = step
            history.append(row)
            if row["nll"] < best_nll:
                best_nll = row["nll"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_wall = time.perf_counter() - start
    if best_state is None:
        raise AssertionError("missing development checkpoint")
    model.load_state_dict(best_state)
    final = eval_model(model, dev, dev_positions)
    final.update({"seed": seed, "condition": condition, "updates": UPDATES,
                  "train_characters_seen": UPDATES * BATCH * BLOCK,
                  "active_macs_per_token_proxy": active_macs_per_token(condition),
                  "train_wall_seconds": train_wall,
                  "inference_payload": serialize(seed, condition, model, vocab, corpus_manifest),
                  "history": history})
    return final


def gates(rows):
    by = {(r["seed"], r["condition"]): r for r in rows}
    checks = []
    for seed in SEEDS:
        mirror = by[(seed, "mirror2x")]
        native = by[(seed, "unipool4")]
        film = by[(seed, "film2x")]
        hard = by[(seed, "hard_alias2x")]
        checks.append({"seed": seed,
            "quality_vs_unipool4": mirror["nll"] <= native["nll"] + .10,
            "whole_payload_saving": mirror["inference_payload"]["bytes"] <= .95 * native["inference_payload"]["bytes"],
            "beats_film": mirror["nll"] <= film["nll"] - .02,
            "beats_hard_alias": mirror["nll"] <= hard["nll"] - .02,
            "film_byte_match": abs(mirror["inference_payload"]["bytes"] - film["inference_payload"]["bytes"]) <= .05 * mirror["inference_payload"]["bytes"],
            "hard_byte_match": abs(mirror["inference_payload"]["bytes"] - hard["inference_payload"]["bytes"]) <= .05 * mirror["inference_payload"]["bytes"],
            "smaller_expert_bank": mirror["inference_payload"]["expert_bank_tensor_bytes"] < native["inference_payload"]["expert_bank_tensor_bytes"]})
    passed = all(all(v for k, v in x.items() if k != "seed") for x in checks)
    return {"development_gate_pass": passed, "checks": checks,
            "audit_eligible": bool(passed), "audit_opening_rule": "only all checks pass in both seeds"}


def evaluate_audit(rows, vocab):
    audit, audit_manifest = open_audit_tokens(vocab)
    measurements = []
    for seed in SEEDS:
        positions = sample_positions(audit.numel(), BATCH * EVAL_BATCHES, seed + 40000)
        for condition in ("unipool4", "hard_alias2x", "mirror2x", "film2x"):
            payload_path = ARTIFACTS / f"seed{seed}_{condition}_inference.pt"
            payload = torch.load(payload_path, map_location="cpu", weights_only=False)
            model = SmallGPT(make_config(), condition, len(vocab))
            model.load_state_dict(payload["model_state"]); model.eval()
            measured = eval_model(model, audit, positions)
            measured.update({"seed": seed, "condition": condition,
                             "serialized_bytes": payload_path.stat().st_size,
                             "serialized_sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest()})
            measurements.append(measured)
    by = {(r["seed"], r["condition"]): r for r in measurements}
    checks = []
    for seed in SEEDS:
        mirror, native = by[(seed, "mirror2x")], by[(seed, "unipool4")]
        film, hard = by[(seed, "film2x")], by[(seed, "hard_alias2x")]
        checks.append({"seed": seed,
            "quality_vs_unipool4": mirror["nll"] <= native["nll"] + .10,
            "beats_film": mirror["nll"] <= film["nll"] - .02,
            "beats_hard_alias": mirror["nll"] <= hard["nll"] - .02})
    passed = all(all(v for k, v in row.items() if k != "seed") for row in checks)
    return {"audit_gate_pass": passed, "checks": checks, "split": audit_manifest,
            "measurements": measurements}


def main():
    torch.set_num_threads(1)
    corpus_manifest = acquire()
    train, dev, vocab = split_and_encode(corpus_manifest)
    results = []
    for seed in SEEDS:
        models = init_models(seed, len(vocab))
        for condition in CONDITIONS:
            print(f"START seed={seed} condition={condition}", flush=True)
            result = train_one(seed, condition, models[condition], train, dev, vocab, corpus_manifest)
            results.append(result)
            print(f"DONE seed={seed} condition={condition} nll={result['nll']:.6f} bytes={result['inference_payload']['bytes']} wall={result['train_wall_seconds']:.1f}s", flush=True)
    gate = gates(results)
    audit = evaluate_audit(results, vocab) if gate["audit_eligible"] else None
    summary = {"corpus": corpus_manifest,
               "split": {"audit_opened": audit is not None, "train_characters": train.numel(), "dev_characters": dev.numel()},
               "results": results, "gates": gate, "audit": audit,
               "failed_attempts": json.loads((SOURCE / "attempt_log.json").read_text()) if (SOURCE / "attempt_log.json").exists() else []}
    out = SOURCE / "development_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    with (ROOT / "RESULTS_CORE.csv").open("w", newline="") as f:
        fields = ["condition", "world_or_seed", "method", "serialized_bytes", "expert_bank_tensor_bytes",
                  "train_tokens_or_examples", "optimizer_updates", "active_compute_proxy", "wall_time_s",
                  "primary_metric", "primary_value", "secondary_metric", "secondary_value", "status_note"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for r in results:
            writer.writerow({"condition": r["condition"], "world_or_seed": r["seed"], "method": "MA-784 frozen screen",
                "serialized_bytes": r["inference_payload"]["bytes"], "expert_bank_tensor_bytes": r["inference_payload"]["expert_bank_tensor_bytes"],
                "train_tokens_or_examples": r["train_characters_seen"], "optimizer_updates": r["updates"],
                "active_compute_proxy": r["active_macs_per_token_proxy"], "wall_time_s": r["train_wall_seconds"],
                "primary_metric": "dev NLL", "primary_value": r["nll"], "secondary_metric": "dev tokens/sec",
                "secondary_value": r["evaluation_tokens_per_second"], "status_note": "audit unopened"})
        if audit is not None:
            for r in audit["measurements"]:
                writer.writerow({"condition": r["condition"], "world_or_seed": f"audit-{r['seed']}", "method": "locked audit",
                    "serialized_bytes": r["serialized_bytes"], "expert_bank_tensor_bytes": "",
                    "train_tokens_or_examples": 0, "optimizer_updates": 0,
                    "active_compute_proxy": active_macs_per_token(r["condition"]), "wall_time_s": r["evaluation_wall_seconds"],
                    "primary_metric": "audit NLL", "primary_value": r["nll"], "secondary_metric": "audit tokens/sec",
                    "secondary_value": r["evaluation_tokens_per_second"], "status_note": "no tuning"})
        for attempt in summary["failed_attempts"]:
            writer.writerow({"condition": attempt["condition"], "world_or_seed": attempt["seed"],
                "method": "failed serialization attempt; development only", "serialized_bytes": "",
                "expert_bank_tensor_bytes": "", "train_tokens_or_examples": attempt["train_tokens_seen"],
                "optimizer_updates": attempt["updates_completed"], "active_compute_proxy": active_macs_per_token(attempt["condition"]),
                "wall_time_s": attempt["observed_process_wall_seconds"], "primary_metric": "not retained",
                "primary_value": "", "secondary_metric": "serialization error", "secondary_value": "",
                "status_note": attempt["failure"]})
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
