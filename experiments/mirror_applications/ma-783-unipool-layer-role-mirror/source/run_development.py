"""Fixed-budget development screen for MA-783. Audit bytes are opened only after gates pass."""
from __future__ import annotations
import csv, hashlib, json, math, statistics, time
from pathlib import Path
import torch
from torch.nn import functional as F
from acquire_corpus import DATA, acquire
from model import CONDITIONS, ModelConfig, SmallGPT, estimate_normrouter_c

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
ARTIFACTS = SOURCE / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
SEEDS = (78301, 78302)
WIDTH, LAYERS, HEADS, EXPERTS, FF = 64, 4, 4, 4, 128
BLOCK, BATCH, UPDATES, EVAL_EVERY, EVAL_BATCHES = 128, 16, 1200, 100, 8
LR, WEIGHT_DECAY, AUX_WEIGHT = 0.0006, 0.1, 0.01
GATE_GROUPS, DEPTH_CODE_DIM = 8, 8


def make_config():
    return ModelConfig(layers=LAYERS, width=WIDTH, heads=HEADS, experts=EXPERTS,
                       expert_hidden=FF, block_size=BLOCK, dropout=0.0,
                       gate_groups=GATE_GROUPS, depth_code_dim=DEPTH_CODE_DIM,
                       aux_weight=AUX_WEIGHT, pool_aux_weight=AUX_WEIGHT,
                       normrouter_c=estimate_normrouter_c(EXPERTS, 1))


def sample_positions(length: int, count: int, seed: int) -> torch.Tensor:
    if length <= BLOCK + 1:
        raise ValueError(f"corpus span too short for block size: {length}")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, length - BLOCK - 1, (count,), generator=gen)


def batch_at(tokens: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.arange(BLOCK + 1)
    seq = tokens[positions[:, None] + offsets[None, :]]
    return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()


def eval_model(model: SmallGPT, tokens: torch.Tensor, positions: torch.Tensor) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loads, entropies = [], []
    start = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, len(positions), BATCH):
            pos = positions[offset:offset + BATCH]
            if not len(pos):
                continue
            x, y = batch_at(tokens, pos)
            logits, _, _, load, entropy = model(x, y)
            count = y.numel()
            total_loss += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum"))
            total_tokens += count
            if load.numel():
                loads.append(load)
                entropies.append(float(entropy))
    elapsed = time.perf_counter() - start
    model.train()
    load = torch.stack(loads).mean(0).tolist() if loads else [0.0] * EXPERTS
    return {"nll": total_loss / total_tokens, "bits_per_character": total_loss / total_tokens / math.log(2),
            "router_load": load, "router_entropy": statistics.mean(entropies) if entropies else 0.0,
            "evaluation_wall_seconds": elapsed,
            "evaluation_tokens_per_second": total_tokens / max(elapsed, 1e-12)}


def copy_common_init(target: SmallGPT, pool: SmallGPT, condition: str):
    src = pool.state_dict()
    dst = target.state_dict()
    common_starts = ("wte.", "wpe.", "lm_head.", "ln_f.")
    for key, value in dst.items():
        if key.startswith(common_starts) or ".ln_1." in key or ".attn." in key or ".ln_2." in key or ".router." in key:
            if key in src and src[key].shape == value.shape:
                value.copy_(src[key])
        if condition in ("unipool", "mirror_givens", "film_gate", "depth_embedding") and key.startswith("shared_experts."):
            value.copy_(src[key])
        if condition == "untied_moe" and key.startswith("layers.") and ".experts." in key:
            tail = key.split(".experts.", 1)[1]
            expert_id = tail.split(".", 1)[0]
            suffix = tail.split(".", 1)[1]
            value.copy_(src[f"shared_experts.{expert_id}.{suffix}"])
        if condition == "dense" and key.startswith("layers.") and ".dense_mlp." in key:
            suffix = key.split(".dense_mlp.", 1)[1]
            value.copy_(src[f"shared_experts.0.{suffix}"])
    target.load_state_dict(dst)


def seeded_models(seed: int, vocab_size: int) -> dict[str, SmallGPT]:
    cfg = make_config()
    torch.manual_seed(seed)
    pool = SmallGPT(cfg, "unipool", vocab_size)
    models = {"unipool": pool}
    for i, condition in enumerate(CONDITIONS):
        if condition == "unipool":
            continue
        torch.manual_seed(seed + 100 + i)
        model = SmallGPT(cfg, condition, vocab_size)
        copy_common_init(model, pool, condition)
        models[condition] = model
    return models


def active_macs_per_token(condition: str) -> int:
    d, t, ff, k, l = WIDTH, BLOCK, FF, EXPERTS, LAYERS
    attn = 4 * d * d + 2 * t * d
    router = d * k + 3 * k if condition != "dense" else 0
    expert = 2 * d * ff
    view = 48 if condition == "mirror_givens" else (d if condition == "film_gate" else 8 * d if condition == "depth_embedding" else 0)
    return l * (attn + router + expert + view)


def save_inference_payload(seed: int, condition: str, model: SmallGPT, vocab: list[str], manifest: dict) -> dict:
    cfg = {"layers": LAYERS, "width": WIDTH, "heads": HEADS, "experts": EXPERTS,
           "expert_hidden": FF, "block_size": BLOCK, "condition": condition,
           "gate_groups": GATE_GROUPS, "depth_code_dim": DEPTH_CODE_DIM,
           "aux_weight": AUX_WEIGHT, "pool_aux_weight": AUX_WEIGHT,
           "normrouter_c": model.config.normrouter_c, "routing": "top1-normrouter"}
    payload = {"schema": "MA-783/inference-v1", "seed": seed, "condition": condition,
               "config": cfg, "vocabulary": vocab,
               "corpus_sha256": manifest["sha256"],
               "model_state": {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}}
    path = ARTIFACTS / f"seed{seed}_{condition}_inference.pt"
    torch.save(payload, path)
    raw = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def measure_role_diversity(model: SmallGPT, tokens: torch.Tensor, positions: torch.Tensor) -> float:
    """Mean pairwise cosine distance of layer expert functions on shared dev activations."""
    captured = [None] * len(model.layers)
    handles = [layer.ln_2.register_forward_hook(lambda _m, _i, out, slot=i: captured.__setitem__(slot, out.detach()))
               for i, layer in enumerate(model.layers)]
    model.eval()
    x, y = batch_at(tokens, positions[:BATCH])
    with torch.no_grad():
        model(x, y)
    for handle in handles:
        handle.remove()
    common = torch.cat([h.reshape(-1, WIDTH) for h in captured], dim=0)[:512]
    outputs = []
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            h = common
            router_input = h
            expert_input = h
            if model.condition == "mirror_givens":
                expert_input = apply_view(h, model.givens[i])
            elif model.condition == "film_gate":
                group_ids = torch.arange(WIDTH) // (WIDTH // GATE_GROUPS)
                expert_input = h * model.film_scale[i][group_ids]
            elif model.condition == "depth_embedding":
                router_input = h + model.depth_projection(model.depth_codes[i]).view(1, -1)
                expert_input = router_input
            if model.condition == "dense":
                out = layer.dense_mlp(expert_input)
            else:
                logits = layer.router(router_input)
                normalized = logits / torch.linalg.vector_norm(logits, dim=-1, keepdim=True).clamp_min(1e-8)
                scores = F.relu(normalized) * model.config.normrouter_c * layer.router_scale
                weights, indices = scores.max(dim=-1)
                bank = model.shared_experts if model.shared_experts is not None else layer.experts
                out = torch.zeros_like(expert_input)
                for expert_id, expert in enumerate(bank):
                    mask = torch.nonzero(indices == expert_id, as_tuple=False).squeeze(-1)
                    if mask.numel():
                        values = expert(expert_input.index_select(0, mask)) * weights.index_select(0, mask).unsqueeze(-1)
                        out = out.index_copy(0, mask, values)
            outputs.append(out)
    distances = []
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            a, b = outputs[i], outputs[j]
            cosine = F.cosine_similarity(a, b, dim=-1, eps=1e-8)
            distances.append(float((1.0 - cosine).mean()))
    return statistics.mean(distances) if distances else 0.0


def apply_view(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    # The model implementation handles arbitrary leading dimensions.
    from model import apply_givens
    return apply_givens(x, angles)


def train_one(seed: int, condition: str, model: SmallGPT, train_tokens: torch.Tensor,
              dev_tokens: torch.Tensor, dev_positions: torch.Tensor, vocab: list[str], manifest: dict) -> dict:
    torch.manual_seed(seed + 10000 + CONDITIONS.index(condition))
    train_gen = torch.Generator(device="cpu").manual_seed(seed + 20000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history = []
    best_nll = float("inf")
    best_state = None
    train_start = time.perf_counter()
    model.train()
    for step in range(1, UPDATES + 1):
        starts = sample_train_positions(train_tokens.numel(), BATCH, train_gen)
        x, y = batch_at(train_tokens, starts)
        _, lm_loss, aux, _, _ = model(x, y)
        loss = lm_loss + AUX_WEIGHT * aux
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % EVAL_EVERY == 0 or step == UPDATES:
            row = eval_model(model, dev_tokens, dev_positions)
            row.update({"step": step, "training_objective": float(loss.detach())})
            history.append(row)
            if row["nll"] < best_nll:
                best_nll = row["nll"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_wall = time.perf_counter() - train_start
    if best_state is None:
        raise AssertionError("no development checkpoint was evaluated")
    model.load_state_dict(best_state)
    final = eval_model(model, dev_tokens, dev_positions)
    final["layer_role_output_cosine_distance"] = measure_role_diversity(model, dev_tokens, dev_positions)
    payload = save_inference_payload(seed, condition, model, vocab, manifest)
    return {"seed": seed, "condition": condition, "best_step": history[min(range(len(history)), key=lambda i: history[i]["nll"])]["step"],
            "development": final, "history": history, "train_wall_seconds": train_wall,
            "train_tokens": UPDATES * BATCH * BLOCK,
            "updates": UPDATES,
            "forward_macs_per_token_proxy": active_macs_per_token(condition),
            "training_forward_backward_macs_proxy": active_macs_per_token(condition) * UPDATES * BATCH * BLOCK * 3,
            "inference_payload": payload}


def sample_train_positions(length: int, batch: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randint(0, length - BLOCK - 1, (batch,), generator=generator)


def run_audit(manifest: dict, models_info: dict, vocab: list[str]) -> dict:
    raw = (DATA / "input.txt").read_bytes()
    start, end = manifest["audit_byte_span_locked"]
    audit_text = raw[start:end].decode("utf-8", errors="replace")
    stoi = {ch: i for i, ch in enumerate(vocab)}
    unk = stoi["\ufffd"]
    audit_tokens = torch.tensor([stoi.get(ch, unk) for ch in audit_text], dtype=torch.int64)
    world_results = {}
    cfg = make_config()
    for seed in SEEDS:
        positions = sample_positions(audit_tokens.numel(), 512, 78303 + seed - SEEDS[0])
        world_results[str(seed)] = {}
        for condition in CONDITIONS:
            model = SmallGPT(cfg, condition, len(vocab))
            payload_path = ROOT / models_info[str(seed)][condition]["inference_payload"]["path"]
            payload = torch.load(payload_path, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model_state"])
            world_results[str(seed)][condition] = eval_model(model, audit_tokens, positions)
    return {"audit_bytes_decoded":len(audit_text),"worlds":world_results,
            "mean_nll_by_condition":{c:statistics.mean(world_results[str(s)][c]["nll"] for s in SEEDS) for c in CONDITIONS}}


def main():
    torch.set_num_threads(1)
    manifest = acquire()
    train_tokens = torch.load(DATA / "train_tokens.pt", map_location="cpu", weights_only=True)
    dev_tokens = torch.load(DATA / "development_tokens.pt", map_location="cpu", weights_only=True)
    vocab = manifest["training_vocabulary"]
    world_positions = {seed: sample_positions(dev_tokens.numel(), EVAL_BATCHES * BATCH, seed + 30000) for seed in SEEDS}
    run_rows, raw_rows, models_info = [], [], {}
    all_start = time.perf_counter()
    for seed in SEEDS:
        models = seeded_models(seed, len(vocab))
        models_info[str(seed)] = {}
        for condition in CONDITIONS:
            result = train_one(seed, condition, models[condition], train_tokens, dev_tokens,
                               world_positions[seed], vocab, manifest)
            models_info[str(seed)][condition] = result
            row = {"world_seed":seed,"evaluation_split":"development","method":condition,"nll":result["development"]["nll"],
                   "bits_per_character":result["development"]["bits_per_character"],
                   "router_load":result["development"]["router_load"],
                   "router_entropy":result["development"]["router_entropy"],
                   "layer_role_output_cosine_distance":result["development"]["layer_role_output_cosine_distance"],
                   "payload_bytes":result["inference_payload"]["bytes"],
                   "payload_sha256":result["inference_payload"]["sha256"],
                   "train_tokens":result["train_tokens"],"updates":result["updates"],
                   "train_wall_seconds":result["train_wall_seconds"],
                   "inference_tokens_per_second":result["development"]["evaluation_tokens_per_second"],
                   "forward_macs_per_token_proxy":result["forward_macs_per_token_proxy"],
                   "training_forward_backward_macs_proxy":result["training_forward_backward_macs_proxy"],
                   "best_step":result["best_step"]}
            raw_rows.append(row)
            run_rows.append((row, result))

    by_seed = {str(seed):{condition:models_info[str(seed)][condition] for condition in CONDITIONS} for seed in SEEDS}
    quality_storage = all(
        by_seed[str(seed)]["mirror_givens"]["development"]["nll"] <= by_seed[str(seed)]["untied_moe"]["development"]["nll"] + 0.10
        and by_seed[str(seed)]["mirror_givens"]["inference_payload"]["bytes"] <= 0.70 * by_seed[str(seed)]["untied_moe"]["inference_payload"]["bytes"]
        for seed in SEEDS)
    mirror_specific = all(
        by_seed[str(seed)]["mirror_givens"]["inference_payload"]["bytes"] * 0.95 <= by_seed[str(seed)][control]["inference_payload"]["bytes"] <= by_seed[str(seed)]["mirror_givens"]["inference_payload"]["bytes"] * 1.05
        and by_seed[str(seed)]["mirror_givens"]["development"]["nll"] <= by_seed[str(seed)][control]["development"]["nll"] - 0.02
        for seed in SEEDS for control in ("film_gate", "depth_embedding"))
    summary = {"experiment_id":"MA-783","status":"DEVELOPMENT_COMPLETE_PENDING_AUDIT_GATE",
               "dataset_manifest":manifest,"dataset_manifest_sha256":hashlib.sha256((DATA/"manifest.json").read_bytes()).hexdigest(),
               "world_seeds":list(SEEDS),"conditions":list(CONDITIONS),"updates_per_run":UPDATES,
               "development_only":True,"audit_accessed":False,"quality_storage_gate_pass":quality_storage,
               "mirror_specific_gate_pass":mirror_specific,"all_development_gates_pass":quality_storage and mirror_specific,
               "decision":"PASS_DEVELOPMENT" if quality_storage and mirror_specific else "FAIL",
               "per_seed":models_info,"total_wall_seconds":time.perf_counter()-all_start}
    (SOURCE/"development_raw.json").write_text(json.dumps(raw_rows,indent=2)+"\n")
    (SOURCE/"development_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    with (ROOT/"RESULTS_CORE.csv").open("w",newline="") as f:
        fields=list(raw_rows[0].keys())
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows(raw_rows)
    if summary["all_development_gates_pass"]:
        summary["audit_accessed"] = True
        audit = run_audit(manifest, models_info, vocab)
        audit_quality = all(
            audit["worlds"][str(seed)]["mirror_givens"]["nll"] <= audit["worlds"][str(seed)]["untied_moe"]["nll"] + 0.10
            and audit["worlds"][str(seed)]["mirror_givens"]["nll"] <= audit["worlds"][str(seed)]["film_gate"]["nll"] - 0.02
            and audit["worlds"][str(seed)]["mirror_givens"]["nll"] <= audit["worlds"][str(seed)]["depth_embedding"]["nll"] - 0.02
            for seed in SEEDS)
        audit["audit_quality_gate_pass"] = audit_quality
        summary["audit_results"] = audit
        summary["decision"] = "PROMISING" if audit_quality else "FAIL"
        audit_rows = []
        for seed in SEEDS:
            for condition in CONDITIONS:
                metric = audit["worlds"][str(seed)][condition]
                train_result = models_info[str(seed)][condition]
                audit_rows.append({"world_seed":seed,"evaluation_split":"audit","method":condition,
                    "nll":metric["nll"],"bits_per_character":metric["bits_per_character"],
                    "router_load":metric["router_load"],"router_entropy":metric["router_entropy"],
                    "layer_role_output_cosine_distance":"not measured on audit",
                    "payload_bytes":train_result["inference_payload"]["bytes"],
                    "payload_sha256":train_result["inference_payload"]["sha256"],
                    "train_tokens":train_result["train_tokens"],"updates":train_result["updates"],
                    "train_wall_seconds":train_result["train_wall_seconds"],
                    "inference_tokens_per_second":metric["evaluation_tokens_per_second"],
                    "forward_macs_per_token_proxy":train_result["forward_macs_per_token_proxy"],
                    "training_forward_backward_macs_proxy":train_result["training_forward_backward_macs_proxy"],
                    "best_step":train_result["best_step"]})
        with (ROOT/"RESULTS_CORE.csv").open("a",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");writer.writerows(audit_rows)
        (SOURCE/"audit_raw.json").write_text(json.dumps(audit_rows,indent=2)+"\n")
        (SOURCE/"development_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


if __name__ == "__main__":
    result=main()
    print(json.dumps({k:result[k] for k in ("decision","quality_storage_gate_pass","mirror_specific_gate_pass","audit_accessed","total_wall_seconds")},indent=2))
