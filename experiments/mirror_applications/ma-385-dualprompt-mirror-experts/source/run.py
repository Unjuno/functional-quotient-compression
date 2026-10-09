"""Sequential task-incremental expert-prompt bank screen for MA-385."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from model import (INPUT_DIM, PROMPT_DIM, RANK, TASKS, ExpertPromptBank,
                   prompt_reconstruction_mac_proxy, selected_prompt_mac_proxy)

torch.set_num_threads(1)
UPDATES_PER_TASK = 1200
BATCH = 64
LR = 0.01
TRAIN_PER_TASK = 1024
TEST_PER_TASK = 1024


def make_world(seed: int) -> dict[str, object]:
    g = torch.Generator().manual_seed(seed)
    a0 = torch.randn(RANK, INPUT_DIM, generator=g) / INPUT_DIM**0.5
    b0 = 0.4 * torch.randn(INPUT_DIM, RANK, generator=g) / RANK**0.5
    j = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    basis = torch.stack(((b0 @ a0).reshape(-1), (b0 @ j @ a0).reshape(-1)), dim=1)
    theta = (2 * torch.rand(TASKS, generator=g) - 1) * np.pi
    coeff = torch.stack((torch.cos(theta), torch.sin(theta)), dim=1)
    expert_values = coeff @ basis.T
    expert_matrices = expert_values.reshape(TASKS, INPUT_DIM, INPUT_DIM)
    general = 0.15 * torch.randn(PROMPT_DIM, generator=g)
    general_matrix = general.reshape(INPUT_DIM, INPUT_DIM)
    data: dict[str, object] = {"general_prompt": general, "teacher_expert": expert_values}
    for split, count in (("train", TRAIN_PER_TASK), ("test", TEST_PER_TASK)):
        x = torch.randn(TASKS, count, INPUT_DIM, generator=g)
        expert_y = torch.einsum("tij,tbj->tbi", expert_matrices, x)
        general_y = torch.einsum("ij,tbj->tbi", general_matrix, x)
        data[f"x_{split}"] = x
        data[f"target_{split}"] = general_y + expert_y
        data[f"expert_target_{split}"] = expert_y
    return data


def nrmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - target) ** 2) / torch.mean(target**2)))


def expert_prompt(bank: ExpertPromptBank, task: int) -> torch.Tensor:
    if bank.method == "independent":
        return bank.expert[task]
    if bank.method == "tied":
        return bank.expert
    if bank.method == "scalar":
        return bank.expert * bank.code[task]
    if bank.method == "coeff":
        return bank.basis @ bank.code[task]
    coords = torch.cat((torch.cos(bank.angle[task]), torch.sin(bank.angle[task])))
    return bank.basis @ coords


def predict_prompt(x: torch.Tensor, general: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
    matrix = (general + prompt).reshape(INPUT_DIM, INPUT_DIM)
    return x @ matrix.T


def stage_metrics(bank: ExpertPromptBank, world: dict[str, object], last_task: int) -> list[float]:
    errors = []
    bank.eval()
    with torch.no_grad():
        for task in range(last_task + 1):
            x = world["x_test"][task]
            ids = torch.full((len(x),), task, dtype=torch.long)
            pred = bank.forward_task(x, ids)
            errors.append(nrmse(pred, world["target_test"][task]))
    return errors


def fit_local(prompt: torch.nn.Parameter, bank: ExpertPromptBank, task: int,
              x: torch.Tensor, target: torch.Tensor, seed: int, updates: int = UPDATES_PER_TASK):
    opt = torch.optim.Adam([prompt], lr=LR)
    gen = torch.Generator().manual_seed(seed)
    start = time.perf_counter()
    for _ in range(updates):
        ids = torch.randint(len(x), (BATCH,), generator=gen)
        pred = predict_prompt(x[ids], bank.general, prompt)
        loss = torch.mean((pred - target[ids]) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return time.perf_counter() - start


def fit_bank(method: str, seed: int, world: dict[str, object]):
    bank = ExpertPromptBank(method, seed + 1000, world["general_prompt"])
    stage_table = []
    total_train_s = 0.0
    if method == "independent":
        for task in range(TASKS):
            local = torch.nn.Parameter(bank.expert[task].detach().clone())
            total_train_s += fit_local(local, bank, task, world["x_train"][task],
                                       world["target_train"][task], seed + 2000 + task)
            with torch.no_grad():
                bank.expert[task].copy_(local)
            errors = stage_metrics(bank, world, task)
            stage_table.append({"after_task": task, "seen_task_nrmse": errors,
                                "mean_seen_nrmse": float(np.mean(errors))})
    elif method == "tied":
        for task in range(TASKS):
            total_train_s += fit_local(bank.expert, bank, task, world["x_train"][task],
                                       world["target_train"][task], seed + 2000 + task)
            errors = stage_metrics(bank, world, task)
            stage_table.append({"after_task": task, "seen_task_nrmse": errors,
                                "mean_seen_nrmse": float(np.mean(errors))})
    elif method == "scalar":
        # Learn one physical expert vector from task 0, then freeze it and add only task gates.
        local = torch.nn.Parameter(bank.expert.detach().clone())
        alpha0 = torch.nn.Parameter(bank.code[0].detach().clone())
        opt = torch.optim.Adam([local, alpha0], lr=LR)
        x0, y0 = world["x_train"][0], world["target_train"][0]
        gen = torch.Generator().manual_seed(seed + 2000)
        start = time.perf_counter()
        for _ in range(UPDATES_PER_TASK):
            ids = torch.randint(len(x0), (BATCH,), generator=gen)
            pred = predict_prompt(x0[ids], bank.general, local * alpha0)
            loss = torch.mean((pred - y0[ids]) ** 2)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        total_train_s += time.perf_counter() - start
        with torch.no_grad():
            bank.expert.copy_(local); bank.code[0].copy_(alpha0)
        errors = stage_metrics(bank, world, 0)
        stage_table.append({"after_task": 0, "seen_task_nrmse": errors,
                            "mean_seen_nrmse": float(np.mean(errors))})
        for task in range(1, TASKS):
            alpha = torch.nn.Parameter(bank.code[task].detach().clone())
            opt = torch.optim.Adam([alpha], lr=LR)
            x, y = world["x_train"][task], world["target_train"][task]
            gen = torch.Generator().manual_seed(seed + 2000 + task)
            start = time.perf_counter()
            for _ in range(UPDATES_PER_TASK):
                ids = torch.randint(len(x), (BATCH,), generator=gen)
                pred = predict_prompt(x[ids], bank.general, bank.expert * alpha)
                loss = torch.mean((pred - y[ids]) ** 2)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total_train_s += time.perf_counter() - start
            with torch.no_grad(): bank.code[task].copy_(alpha)
            errors = stage_metrics(bank, world, task)
            stage_table.append({"after_task": task, "seen_task_nrmse": errors,
                                "mean_seen_nrmse": float(np.mean(errors))})
    else:
        # The first two tasks identify a shared 2D expert basis; later tasks only add a new code.
        code_parameter = bank.code if method == "coeff" else bank.angle
        optimizer = torch.optim.Adam([bank.basis, code_parameter], lr=LR)
        x0, x1 = world["x_train"][0], world["x_train"][1]
        y0, y1 = world["target_train"][0], world["target_train"][1]
        g0 = torch.Generator().manual_seed(seed + 2000)
        g1 = torch.Generator().manual_seed(seed + 2001)
        start = time.perf_counter()
        for _ in range(UPDATES_PER_TASK):
            i0 = torch.randint(len(x0), (BATCH,), generator=g0)
            i1 = torch.randint(len(x1), (BATCH,), generator=g1)
            p0, p1 = expert_prompt(bank, 0), expert_prompt(bank, 1)
            loss = (torch.mean((predict_prompt(x0[i0], bank.general, p0) - y0[i0]) ** 2) +
                    torch.mean((predict_prompt(x1[i1], bank.general, p1) - y1[i1]) ** 2)) / 2
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        total_train_s += time.perf_counter() - start
        errors = stage_metrics(bank, world, 1)
        stage_table.append({"after_task": 1, "seen_task_nrmse": errors,
                            "mean_seen_nrmse": float(np.mean(errors))})
        bank.basis.requires_grad_(False)
        for task in range(2, TASKS):
            if method == "coeff":
                local_code = torch.nn.Parameter(bank.code[task].detach().clone())
            else:
                local_code = torch.nn.Parameter(bank.angle[task].detach().clone())
            opt = torch.optim.Adam([local_code], lr=LR)
            x, y = world["x_train"][task], world["target_train"][task]
            gen = torch.Generator().manual_seed(seed + 2000 + task)
            start = time.perf_counter()
            for _ in range(UPDATES_PER_TASK):
                ids = torch.randint(len(x), (BATCH,), generator=gen)
                if method == "coeff":
                    prompt = bank.basis @ local_code
                else:
                    prompt = bank.basis @ torch.cat((torch.cos(local_code), torch.sin(local_code)))
                pred = predict_prompt(x[ids], bank.general, prompt)
                loss = torch.mean((pred - y[ids]) ** 2)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total_train_s += time.perf_counter() - start
            with torch.no_grad():
                if method == "coeff": bank.code[task].copy_(local_code)
                else: bank.angle[task].copy_(local_code)
            errors = stage_metrics(bank, world, task)
            stage_table.append({"after_task": task, "seen_task_nrmse": errors,
                                "mean_seen_nrmse": float(np.mean(errors))})
    final_errors = stage_table[-1]["seen_task_nrmse"]
    learned_errors = []
    for task in range(TASKS):
        completed_stage = max(task, 1) if method in ("coeff", "mirror") else task
        entry = next(e for e in stage_table if e["after_task"] == completed_stage)
        learned_errors.append(entry["seen_task_nrmse"][task])
    forgetting = [max(0.0, final_errors[i] - learned_errors[i]) for i in range(TASKS)]
    return bank.eval(), {
        "stage_metrics": stage_table,
        "final_task_nrmse": final_errors,
        "final_mean_seen_nrmse": float(np.mean(final_errors)),
        "mean_old_task_forgetting_nrmse": float(np.mean(forgetting[:-1])),
        "per_task_forgetting_nrmse": forgetting,
        "optimizer_updates": UPDATES_PER_TASK * (TASKS if method in ("independent", "tied", "scalar") else TASKS - 1),
        "source_examples_seen": UPDATES_PER_TASK * BATCH * TASKS,
        "training_wall_s": total_train_s,
    }


def save_payload(path: Path, bank: ExpertPromptBank) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **bank.payload_arrays())
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def run(seed: int, condition: str, outdir: Path) -> dict[str, object]:
    world = make_world(seed)
    rows = []
    for method in ExpertPromptBank.METHODS:
        start = time.perf_counter()
        bank, metrics = fit_bank(method, seed, world)
        path = outdir / f"{condition}_{seed}_{method}.npz"
        size, digest = save_payload(path, bank)
        with torch.no_grad():
            x = world["x_test"].reshape(-1, INPUT_DIM)
            ids = torch.arange(TASKS).repeat_interleave(TEST_PER_TASK)
            t0 = time.perf_counter()
            _ = bank.forward_task(x, ids)
            infer_s = time.perf_counter() - t0
        rows.append({
            "method": method, "payload_path": path.name,
            "serialized_bytes": size, "payload_sha256": digest,
            **metrics,
            "prompt_reconstruction_mac_proxy_per_bank": prompt_reconstruction_mac_proxy(method),
            "selected_prompt_mac_proxy_per_query": selected_prompt_mac_proxy(),
            "optimizer_updates_per_task": UPDATES_PER_TASK,
            "inference_examples_per_s": len(x) / max(infer_s, 1e-9),
            "total_wall_s": time.perf_counter() - start,
        })
    return {"experiment_id": "MA-385", "condition": condition, "seed": seed,
            "tasks": TASKS, "updates_per_task": UPDATES_PER_TASK,
            "batch_size": BATCH, "learning_rate": LR, "results": rows}


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
                      "results": [{k: r[k] for k in ("method", "serialized_bytes", "final_mean_seen_nrmse", "mean_old_task_forgetting_nrmse", "total_wall_s")}
                                  for r in result["results"]]}, indent=2))


if __name__ == "__main__":
    main()
