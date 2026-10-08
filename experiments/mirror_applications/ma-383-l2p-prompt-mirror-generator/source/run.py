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

TASKS, KEY_DIM, FEATURE_DIM = 4, 8, 8
BATCH, UPDATES, LR = 128, 600, 3e-3
METHODS = ("independent", "hard_shared", "rank2_coeff", "hypernet", "mirror_angle")
torch.set_num_threads(1)


def make_data(seed):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((TASKS, KEY_DIM)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    # Four well-separated but noisy task signatures; retrieval must still use input features.
    angles = np.linspace(-0.72, 0.72, TASKS) + rng.normal(0, 0.04, TASKS)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    out = {"angles": angles.astype(np.float32), "directions": directions}
    for split, n, salt in (("train", 2048, 11), ("validation", 512, 23), ("test", 1024, 37)):
        srng = np.random.default_rng(seed + salt)
        task_ids = np.repeat(np.arange(TASKS), n)
        xkey = centers[task_ids] + 0.16 * srng.standard_normal((len(task_ids), KEY_DIM))
        z = srng.standard_normal((len(task_ids), FEATURE_DIM)).astype(np.float32)
        score = (z[:, :2] * directions[task_ids]).sum(axis=1)
        y = (score >= 0).astype(np.float32)
        out[split] = {
            "key": torch.tensor(xkey, dtype=torch.float32),
            "features": torch.tensor(z, dtype=torch.float32),
            "labels": torch.tensor(y, dtype=torch.float32),
            "task": torch.tensor(task_ids, dtype=torch.long),
        }
    # L2P-style key prototypes are fitted from training observations, then serialized and reused.
    train = out["train"]
    keys = torch.stack([train["key"][train["task"] == j].mean(dim=0) for j in range(TASKS)])
    keys = F.normalize(keys, dim=1).numpy()
    out["keys"] = keys
    return out


class PromptModel(torch.nn.Module):
    def __init__(self, method, seed):
        super().__init__()
        self.method = method
        torch.manual_seed(seed + METHODS.index(method) * 101)
        if method == "independent":
            self.prompts = torch.nn.Parameter(torch.randn(TASKS, FEATURE_DIM) * 0.05)
        elif method == "hard_shared":
            self.prompt = torch.nn.Parameter(torch.randn(FEATURE_DIM) * 0.05)
        elif method == "rank2_coeff":
            self.basis = torch.nn.Parameter(torch.randn(FEATURE_DIM, 2) * 0.05)
            self.coefficients = torch.nn.Parameter(torch.randn(TASKS, 2) * 0.05)
        elif method == "mirror_angle":
            self.basis = torch.nn.Parameter(torch.randn(FEATURE_DIM, 2) * 0.05)
            self.angles = torch.nn.Parameter(torch.linspace(-0.4, 0.4, TASKS))
        elif method == "hypernet":
            self.generator = torch.nn.Sequential(
                torch.nn.Linear(KEY_DIM, 16), torch.nn.Tanh(),
                torch.nn.Linear(16, FEATURE_DIM),
            )
        else:
            raise ValueError(method)

    def logical_prompts(self, keys):
        if self.method == "independent":
            return self.prompts
        if self.method == "hard_shared":
            return self.prompt[None, :].expand(TASKS, -1)
        if self.method == "rank2_coeff":
            return self.coefficients @ self.basis.T
        if self.method == "mirror_angle":
            coeff = torch.stack([torch.cos(self.angles), torch.sin(self.angles)], dim=1)
            return coeff @ self.basis.T
        return self.generator(keys)

    def forward(self, key, features, keys):
        task = torch.cdist(key, keys).argmin(dim=1)
        prompts = self.logical_prompts(keys)
        logits = (prompts[task] * features).sum(dim=1)
        return logits, task


def train(method, seed, data):
    model = PromptModel(method, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    keys = torch.tensor(data["keys"], dtype=torch.float32)
    start = time.perf_counter()
    # Sequential task blocks are fixed across methods; the same updates/data schedule is paid.
    per_task = UPDATES // TASKS
    for task_id in range(TASKS):
        rows = torch.where(data["train"]["task"] == task_id)[0]
        rng = np.random.default_rng(seed * 13 + task_id)
        for _ in range(per_task):
            idx = torch.from_numpy(rng.choice(rows.numpy(), size=BATCH, replace=True))
            batch = data["train"]
            logits, _ = model(batch["key"][idx], batch["features"][idx], keys)
            loss = F.binary_cross_entropy_with_logits(logits, batch["labels"][idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    return model, time.perf_counter() - start


def evaluate(model, split, keys):
    with torch.no_grad():
        logits, chosen = model(split["key"], split["features"], keys)
        pred = (logits >= 0).float()
        correct = pred.eq(split["labels"])
        per_task = []
        for task_id in range(TASKS):
            mask = split["task"] == task_id
            per_task.append(float(correct[mask].float().mean()))
        return {
            "accuracy": float(correct.float().mean()),
            "bce": float(F.binary_cross_entropy_with_logits(logits, split["labels"])),
            "retrieval_accuracy": float(chosen.eq(split["task"]).float().mean()),
            "per_task_accuracy": per_task,
        }


def state_arrays(model, data):
    arrays = {"router_keys": np.asarray(data["keys"], dtype="<f2"),
              "meta": np.asarray([TASKS, KEY_DIM, FEATURE_DIM, METHODS.index(model.method)], dtype=np.uint16)}
    for k, value in model.state_dict().items():
        arrays[k] = value.detach().cpu().numpy().astype("<f2")
    return arrays


def deterministic_pack(arrays, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buf.getvalue())
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def run(seed, condition, outdir, result_path):
    data = make_data(seed)
    keys = torch.tensor(data["keys"], dtype=torch.float32)
    rows = []
    for method in METHODS:
        model, wall = train(method, seed, data)
        payload = Path(outdir) / f"{condition}_{seed}_{method}.npz"
        arrays = state_arrays(model, data)
        size, digest = deterministic_pack(arrays, payload)
        loaded = PromptModel(method, seed)
        with zipfile.ZipFile(payload) as archive:
            loaded_arrays = {name[:-4]: np.lib.format.read_array(io.BytesIO(archive.read(name)), allow_pickle=False)
                             for name in archive.namelist()}
        state = {k: torch.tensor(loaded_arrays[k], dtype=v.dtype) for k, v in loaded.state_dict().items()}
        loaded.load_state_dict(state)
        loaded.eval()
        inference_keys = torch.tensor(loaded_arrays["router_keys"].astype(np.float32))
        rows.append({
            "method": method, "serialized_bytes": size, "payload_sha256": digest,
            "optimizer_updates": UPDATES, "training_examples_seen": UPDATES * BATCH,
            "train_wall_s": wall, "validation": evaluate(loaded, data["validation"], inference_keys),
            "test": evaluate(loaded, data["test"], inference_keys),
            "macs_per_example_estimate": {
                "retrieval": TASKS * KEY_DIM,
                "prompt_generation": FEATURE_DIM * (2 if method == "mirror_angle" else 1) if method in ("mirror_angle", "rank2_coeff") else (KEY_DIM * 16 + 16 * FEATURE_DIM if method == "hypernet" else 0),
                "score": FEATURE_DIM,
            },
        })
    result = {"condition": condition, "seed": seed, "world_directions": data["directions"].tolist(), "summaries": rows}
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
