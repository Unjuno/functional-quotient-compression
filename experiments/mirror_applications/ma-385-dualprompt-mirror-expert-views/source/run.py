import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TASKS, KEY_DIM, FEATURE_DIM = 8, 8, 16
BATCH, UPDATES, LR = 128, 800, 2e-3
METHODS = ("dualprompt_explicit", "hard_shared", "rank2_coeff", "mirror_angle")
torch.set_num_threads(1)


def make_data(seed):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((TASKS, KEY_DIM)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    general = rng.standard_normal(FEATURE_DIM).astype(np.float32)
    general[8:] = 0.0
    general = 0.85 * general / np.linalg.norm(general)
    phase = float(rng.uniform(-0.2, 0.2))
    angles = np.linspace(0, 2 * np.pi, TASKS, endpoint=False) + phase
    residuals = np.zeros((TASKS, FEATURE_DIM), dtype=np.float32)
    residuals[:, 8] = 0.55 * np.cos(angles)
    residuals[:, 9] = 0.55 * np.sin(angles)
    teachers = general[None, :] + residuals
    out = {"centers": centers, "general_teacher": general, "residual_teachers": residuals}
    for split, n, salt in (("train", 1024, 11), ("validation", 256, 23), ("test", 512, 37)):
        srng = np.random.default_rng(seed + salt)
        task_ids = np.repeat(np.arange(TASKS), n)
        key = centers[task_ids] + 0.16 * srng.standard_normal((len(task_ids), KEY_DIM))
        features = srng.standard_normal((len(task_ids), FEATURE_DIM)).astype(np.float32)
        score = (features * teachers[task_ids]).sum(axis=1)
        out[split] = {
            "key": torch.tensor(key, dtype=torch.float32),
            "features": torch.tensor(features, dtype=torch.float32),
            "labels": torch.tensor((score >= 0).astype(np.float32)),
            "task": torch.tensor(task_ids, dtype=torch.long),
        }
    train = out["train"]
    prototypes = torch.stack([train["key"][train["task"] == j].mean(0) for j in range(TASKS)])
    out["router_keys"] = F.normalize(prototypes, dim=1).numpy()
    return out


class DualPromptModel(torch.nn.Module):
    def __init__(self, method, seed):
        super().__init__()
        self.method = method
        torch.manual_seed(seed + METHODS.index(method) * 131)
        self.general = torch.nn.Parameter(torch.zeros(FEATURE_DIM))
        if method == "dualprompt_explicit":
            self.experts = torch.nn.Parameter(torch.randn(TASKS, FEATURE_DIM) * 0.03)
        elif method == "hard_shared":
            self.expert = torch.nn.Parameter(torch.randn(FEATURE_DIM) * 0.03)
        elif method == "rank2_coeff":
            self.basis = torch.nn.Parameter(torch.randn(FEATURE_DIM, 2) * 0.03)
            self.coefficients = torch.nn.Parameter(torch.randn(TASKS, 2) * 0.03)
        elif method == "mirror_angle":
            self.basis = torch.nn.Parameter(torch.randn(FEATURE_DIM, 2) * 0.03)
            self.angles = torch.nn.Parameter(torch.linspace(-0.5, 0.5, TASKS))
        else:
            raise ValueError(method)

    def prompts(self):
        if self.method == "dualprompt_explicit":
            experts = self.experts
        elif self.method == "hard_shared":
            experts = self.expert[None, :].expand(TASKS, -1)
        elif self.method == "rank2_coeff":
            experts = self.coefficients @ self.basis.T
        else:
            coeff = torch.stack([torch.cos(self.angles), torch.sin(self.angles)], dim=1)
            experts = coeff @ self.basis.T
        return self.general[None, :] + experts

    def forward(self, key, features, router_keys):
        task = torch.cdist(key, router_keys).argmin(1)
        logits = (self.prompts()[task] * features).sum(1)
        return logits, task


def evaluate(model, split, router_keys, max_task=None):
    mask = torch.ones_like(split["task"], dtype=torch.bool)
    if max_task is not None:
        mask = split["task"] <= max_task
    with torch.no_grad():
        logits, chosen = model(split["key"][mask], split["features"][mask], router_keys)
        labels, task_ids = split["labels"][mask], split["task"][mask]
        correct = (logits >= 0).float().eq(labels)
        per_task = []
        for task_id in torch.unique(task_ids, sorted=True):
            per_task.append(float(correct[task_ids == task_id].float().mean()))
        return {
            "accuracy": float(correct.float().mean()),
            "bce": float(F.binary_cross_entropy_with_logits(logits, labels)),
            "retrieval_accuracy": float(chosen.eq(task_ids).float().mean()),
            "per_task_accuracy": per_task,
        }


def arrays_for(model, data):
    arrays = {"router_keys": np.asarray(data["router_keys"], dtype="<f2"),
              "meta": np.asarray([TASKS, KEY_DIM, FEATURE_DIM, METHODS.index(model.method)], dtype=np.uint16)}
    for key, value in model.state_dict().items():
        arrays[key] = value.detach().cpu().numpy().astype("<f2")
    return arrays


def deterministic_pack(arrays, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def write_model_payload(model, data, path):
    arrays = arrays_for(model, data)
    size, digest = deterministic_pack(arrays, path)
    return arrays, size, digest


def load_serialized_model(method, seed, arrays):
    model = DualPromptModel(method, seed)
    model.load_state_dict({k: torch.tensor(arrays[k], dtype=v.dtype) for k, v in model.state_dict().items()})
    model.eval()
    keys = torch.tensor(arrays["router_keys"].astype(np.float32))
    return model, keys


def train(method, seed, data, result_dir):
    model = DualPromptModel(method, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    key_train = data["train"]["key"]
    keys = torch.tensor(data["router_keys"], dtype=torch.float32)
    per_task = UPDATES // TASKS
    history = []
    train_wall = 0.0
    for task_id in range(TASKS):
        indices = torch.where(data["train"]["task"] == task_id)[0]
        rng = np.random.default_rng(seed * 131 + task_id)
        block_start = time.perf_counter()
        for _ in range(per_task):
            idx = torch.from_numpy(rng.choice(indices.numpy(), BATCH, replace=True))
            logits, _ = model(key_train[idx], data["train"]["features"][idx], keys)
            loss = F.binary_cross_entropy_with_logits(logits, data["train"]["labels"][idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        train_wall += time.perf_counter() - block_start
        snapshot = Path(result_dir) / "snapshots" / f"development_{seed}_{method}_seen_{task_id + 1}.npz"
        snapshot_arrays, size, digest = write_model_payload(model, data, snapshot)
        snapshot_model, snapshot_keys = load_serialized_model(method, seed, snapshot_arrays)
        val = evaluate(snapshot_model, data["validation"], snapshot_keys, max_task=task_id)
        history.append({"tasks_seen": task_id + 1, "validation": val,
                        "diagnostic_snapshot_bytes": size, "diagnostic_snapshot_sha256": digest})
    final_payload = Path(result_dir) / "payloads" / f"development_{seed}_{method}.npz"
    arrays, size, digest = write_model_payload(model, data, final_payload)
    loaded, inference_keys = load_serialized_model(method, seed, arrays)
    return {
        "method": method,
        "serialized_bytes": size,
        "payload_sha256": digest,
        "optimizer_updates": UPDATES,
        "training_examples_seen": UPDATES * BATCH,
        "train_wall_s": train_wall,
        "validation": evaluate(loaded, data["validation"], inference_keys),
        "test": evaluate(loaded, data["test"], inference_keys),
        "sequential_validation_history": history,
        "macs_per_example_estimate": {
            "retrieval": TASKS * KEY_DIM,
            "prompt_generation": FEATURE_DIM * 2 if method in ("rank2_coeff", "mirror_angle") else 0,
            "classification": FEATURE_DIM,
        },
    }


def run(seed, condition, output_dir, result_path):
    data = make_data(seed)
    rows = [train(method, seed, data, output_dir) for method in METHODS]
    result = {"condition": condition, "seed": seed,
              "general_teacher": data["general_teacher"].tolist(),
              "residual_teachers": data["residual_teachers"].tolist(), "summaries": rows}
    Path(result_path).parent.mkdir(parents=True, exist_ok=True)
    Path(result_path).write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", choices=["development", "fresh"], required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    run(args.seed, args.condition, args.outdir, args.json)
