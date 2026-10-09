"""CPU prompt-pool and nearest-key retrieval screen for MA-383."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from model import (INPUT_DIM, KEY_DIM, PROMPTS, PromptBank, nearest_key,
                   reconstruction_mac_proxy, retrieval_mac_proxy,
                   selected_prompt_mac_proxy, select_prompt_outputs)

torch.set_num_threads(1)
SOURCE_UPDATES = 1200
BATCH = 64
LR = 0.01
TEST_PER_PROMPT = 1024


def make_world(seed: int) -> dict[str, object]:
    g = torch.Generator().manual_seed(seed)
    a0 = torch.randn(2, INPUT_DIM, generator=g) / INPUT_DIM**0.5
    b0 = 0.4 * torch.randn(INPUT_DIM, 2, generator=g) / 2**0.5
    j = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    basis = torch.stack(((b0 @ a0).reshape(-1), (b0 @ j @ a0).reshape(-1)), dim=1)
    angles = (2 * torch.rand(PROMPTS, generator=g) - 1) * np.pi
    coeff = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    prompt_values = coeff @ basis.T
    matrices = prompt_values.reshape(PROMPTS, INPUT_DIM, INPUT_DIM)
    keys = torch.linalg.qr(torch.randn(KEY_DIM, KEY_DIM, generator=g)).Q.T
    data: dict[str, object] = {"keys": keys, "teacher_prompts": prompt_values,
                               "teacher_matrices": matrices}
    x_train = torch.randn(4096, INPUT_DIM, generator=g)
    data["x_train"] = x_train
    data["source_train"] = torch.einsum("bd,ndk->bnk", x_train, matrices.transpose(1, 2))
    x_validation = torch.randn(1024, INPUT_DIM, generator=g)
    data["x_validation"] = x_validation
    data["source_validation"] = torch.einsum("bd,ndk->bnk", x_validation, matrices.transpose(1, 2))
    x_task = torch.randn(PROMPTS, TEST_PER_PROMPT, INPUT_DIM, generator=g)
    q_task = keys[:, None, :] + 0.35 * torch.randn(PROMPTS, TEST_PER_PROMPT, KEY_DIM, generator=g)
    flat_x = x_task.reshape(-1, INPUT_DIM)
    target = torch.einsum("nij,nbj->nbi", matrices, x_task)
    # [task, example, output]; equivalent to x @ M_task.T.
    data.update({"x_test_task": x_task, "query_test_task": q_task, "target_test_task": target,
                 "x_test": flat_x, "query_test": q_task.reshape(-1, KEY_DIM),
                 "target_test": target.reshape(-1, INPUT_DIM)})
    return data


def nrmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - target) ** 2) / torch.mean(target**2)))


def fit_bank(method: str, seed: int, world: dict[str, object]):
    bank = PromptBank(method, seed + 1000)
    opt = torch.optim.Adam(bank.parameters(), lr=LR)
    x, target = world["x_train"], world["source_train"]
    generator = torch.Generator().manual_seed(seed + 2000)
    start = time.perf_counter()
    for _ in range(SOURCE_UPDATES):
        ids = torch.randint(len(x), (BATCH,), generator=generator)
        pred = bank(x[ids])
        loss = torch.mean((pred - target[ids]) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    elapsed = time.perf_counter() - start
    bank.eval()
    with torch.no_grad():
        source_val = bank(world["x_validation"])
        source_test = bank(world["x_test"])
        val_nrmse = nrmse(source_val, world["source_validation"])
        true_ids = torch.arange(PROMPTS).repeat_interleave(TEST_PER_PROMPT)
        source_test_nrmse = nrmse(select_prompt_outputs(source_test, true_ids), world["target_test"])
    return bank, {"source_validation_nrmse": val_nrmse,
                  "source_test_nrmse": source_test_nrmse}, elapsed


def retrieval_metrics(bank: PromptBank, keys: torch.Tensor, world: dict[str, object]):
    x = world["x_test"]
    query = world["query_test"]
    ids = nearest_key(query, keys)
    true_ids = torch.arange(PROMPTS).repeat_interleave(TEST_PER_PROMPT)
    retrieval_accuracy = float((ids == true_ids).float().mean())
    with torch.no_grad():
        pred = bank.forward_selected(x, ids)
        oracle = bank.forward_selected(x, true_ids)
        target = world["target_test"]
        by_task = []
        pred_task = pred.reshape(PROMPTS, TEST_PER_PROMPT, INPUT_DIM)
        target_task = world["target_test_task"]
        for i in range(PROMPTS):
            by_task.append(nrmse(pred_task[i], target_task[i]))
        return {
            "retrieval_top1_accuracy": retrieval_accuracy,
            "retrieved_test_nrmse_by_prompt": by_task,
            "retrieved_test_nrmse": float(np.mean(by_task)),
            "oracle_key_test_nrmse": nrmse(oracle, target),
            "retrieved_examples": len(x),
        }


def save_payload(path: Path, bank: PromptBank, keys: torch.Tensor) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **bank.payload_arrays(keys))
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def run(seed: int, condition: str, outdir: Path) -> dict[str, object]:
    world = make_world(seed)
    rows = []
    for method in PromptBank.METHODS:
        start = time.perf_counter()
        bank, source_metrics, train_s = fit_bank(method, seed, world)
        infer_metrics = retrieval_metrics(bank, world["keys"], world)
        path = outdir / f"{condition}_{seed}_{method}.npz"
        size, digest = save_payload(path, bank, world["keys"])
        with torch.no_grad():
            x, q, keys = world["x_test"], world["query_test"], world["keys"]
            t0 = time.perf_counter()
            prompt_ids = nearest_key(q, keys)
            _ = bank.forward_selected(x, prompt_ids)
            infer_s = time.perf_counter() - t0
        rows.append({
            "method": method, "payload_path": path.name,
            "serialized_bytes": size, "payload_sha256": digest,
            **source_metrics, **infer_metrics,
            "prompt_reconstruction_mac_proxy_per_bank": reconstruction_mac_proxy(method),
            "retrieval_mac_proxy_per_query": retrieval_mac_proxy(),
            "selected_prompt_mac_proxy_per_query": selected_prompt_mac_proxy(),
            "optimizer_updates": SOURCE_UPDATES,
            "source_examples_seen": SOURCE_UPDATES * BATCH * PROMPTS,
            "inference_examples_per_s": len(x) / max(infer_s, 1e-9),
            "training_wall_s": train_s,
            "total_wall_s": time.perf_counter() - start,
        })
    return {"experiment_id": "MA-383", "condition": condition, "seed": seed,
            "source_updates": SOURCE_UPDATES, "batch_size": BATCH,
            "learning_rate": LR, "test_examples_per_prompt": TEST_PER_PROMPT,
            "retrieval_noise_std": 0.35, "results": rows}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--condition", choices=("development", "fresh"), required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--json", type=Path, required=True)
    args = p.parse_args()
    result = run(args.seed, args.condition, args.outdir)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"seed": args.seed, "condition": args.condition,
                      "results": [{k: r[k] for k in ("method", "serialized_bytes", "retrieval_top1_accuracy", "source_test_nrmse", "retrieved_test_nrmse")}
                                  for r in result["results"]]}, indent=2))


if __name__ == "__main__":
    main()
