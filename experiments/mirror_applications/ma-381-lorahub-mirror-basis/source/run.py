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

from model import (INPUT_DIM, METHODS, OUTPUT_DIM, RANK, SOURCE_TASKS, TARGET_TASKS,
                   CandidateBank, givens)

torch.set_num_threads(1)
UPDATES, BATCH, LR = 1600, 64, 1e-3
RIDGE = 1e-5


def teacher_bank(seed, condition):
    g = torch.Generator().manual_seed(seed + (41000 if condition == "aligned" else 42000))
    if condition == "aligned":
        base_a = torch.randn(INPUT_DIM, RANK, generator=g) * 0.16
        base_b = torch.randn(RANK, OUTPUT_DIM, generator=g) * 0.16
        base = base_a @ base_b
        angles = torch.arange(SOURCE_TASKS, dtype=torch.float32) * 0.19
        matrices = torch.stack([givens(a) @ base @ givens(a).T for a in angles])
    else:
        a = torch.randn(SOURCE_TASKS, INPUT_DIM, RANK, generator=g) * 0.16
        b = torch.randn(SOURCE_TASKS, RANK, OUTPUT_DIM, generator=g) * 0.16
        matrices = torch.einsum("kir,kro->kio", a, b)
    mix = torch.zeros(TARGET_TASKS, SOURCE_TASKS)
    for t in range(TARGET_TASKS):
        mix[t, t] = 1.0
        mix[t, (t + 1) % SOURCE_TASKS] = -0.45
        mix[t, (t + 4) % SOURCE_TASKS] = 0.30
        mix[t, (t + 5) % SOURCE_TASKS] = 0.15
    target_matrices = torch.einsum("tk,kio->tio", mix, matrices)
    return matrices, target_matrices, mix


def make_data(seed, condition):
    sources, targets, mix = teacher_bank(seed, condition)
    result = {}
    for name, n, salt in (("train", 8192, 101), ("validation", 2048, 211), ("test", 4096, 307), ("support", 32, 401)):
        g = torch.Generator().manual_seed(seed + salt)
        x = torch.randn(n, INPUT_DIM, generator=g)
        source_y = torch.einsum("bi,kio->bko", x, sources)
        target_y = torch.einsum("bi,tio->bto", x, targets)
        result[name] = (x, source_y, target_y)
    return result, mix


def train_source(method, seed, condition, train_data):
    torch.manual_seed(seed * 37 + METHODS.index(method))
    offset = 4100 if condition == "aligned" else 8200
    model = CandidateBank(method, seed * 67 + METHODS.index(method) + offset)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    x, source_y, _ = train_data
    rng = np.random.default_rng(seed * 100003 + (15101 if condition == "aligned" else 25101))
    start = time.perf_counter()
    for _ in range(UPDATES):
        ids = torch.from_numpy(rng.integers(0, len(x), size=BATCH, dtype=np.int64))
        loss = F.mse_loss(model.source_outputs(x[ids]), source_y[ids])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    wall = time.perf_counter() - start
    return model.eval(), wall


def fit_fusion(model, support):
    x, _, target_y = support
    with torch.no_grad():
        sources = model.source_outputs(x)
        for task in range(TARGET_TASKS):
            design = sources[:, :, :].permute(0, 2, 1).reshape(-1, SOURCE_TASKS)
            target = target_y[:, task, :].reshape(-1)
            gram = design.T @ design + RIDGE * torch.eye(SOURCE_TASKS)
            rhs = design.T @ target
            model.fusion[task].copy_(torch.linalg.solve(gram, rhs))
    return TARGET_TASKS * (len(x) * OUTPUT_DIM * SOURCE_TASKS**2 + SOURCE_TASKS**3)


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


def payload_arrays(path):
    arrays = {}
    with zipfile.ZipFile(path, "r") as archive:
        for member in archive.namelist():
            arrays[member[:-4]] = np.lib.format.read_array(io.BytesIO(archive.read(member)), allow_pickle=False)
    return arrays


def save_payload(model, method, condition, path):
    arrays = {}
    for key, tensor in model.state_dict().items():
        value = tensor.detach().cpu().numpy()
        arrays[key] = value.astype("<f2") if value.dtype.kind == "f" else value
    arrays["source_ids"] = np.arange(SOURCE_TASKS, dtype=np.uint8)
    arrays["target_ids"] = np.arange(TARGET_TASKS, dtype=np.uint8)
    arrays["meta"] = np.asarray([INPUT_DIM, OUTPUT_DIM, RANK, SOURCE_TASKS, TARGET_TASKS], dtype=np.uint16)
    arrays["method_id"] = np.asarray([METHODS.index(method)], dtype=np.uint8)
    arrays["condition_id"] = np.asarray([0 if condition == "aligned" else 1], dtype=np.uint8)
    return deterministic_pack(arrays, path)


def load_model(method, arrays, seed, condition):
    offset = 4100 if condition == "aligned" else 8200
    model = CandidateBank(method, seed * 67 + METHODS.index(method) + offset)
    state = {}
    for key, value in model.state_dict().items():
        tensor = torch.from_numpy(np.array(arrays[key], copy=True))
        state[key] = tensor.to(value.dtype) if value.is_floating_point() else tensor
    model.load_state_dict(state)
    return model.eval()


def evaluate(model, split, condition):
    x, source_targets, target_targets = split
    with torch.no_grad():
        source = model.source_outputs(x)
        source_mses = ((source - source_targets) ** 2).mean(dim=(0, 2)).tolist()
        start = time.perf_counter()
        prediction = model.target_outputs(x)
        elapsed = time.perf_counter() - start
        target_mses = ((prediction - target_targets) ** 2).mean(dim=(0, 2)).tolist()
    decode_macs = 0
    if model.method == "mirror_shared":
        decode_macs = SOURCE_TASKS * 3 * INPUT_DIM**3
    candidate_macs = SOURCE_TASKS * INPUT_DIM * RANK * 2 + SOURCE_TASKS * OUTPUT_DIM
    return {"source_mses": source_mses, "target_mses": target_mses,
            "mean_source_mse": float(np.mean(source_mses)), "mean_target_mse": float(np.mean(target_mses)),
            "target_examples_per_s": float(len(x) * TARGET_TASKS / max(elapsed, 1e-9)),
            "candidate_encode_macs_per_example": candidate_macs,
            "one_time_view_decode_macs": decode_macs,
            "fusion_coefficients": model.fusion.detach().cpu().tolist(),
            "source_matrix_norms": model.source_matrices().detach().norm(dim=(1, 2)).cpu().tolist()}


def run(seed, split_name, outdir, json_path):
    rows = []
    target_mixes = {}
    for condition in ("aligned", "unrelated"):
        data, mix = make_data(seed, condition)
        target_mixes[condition] = mix.tolist()
        for method in METHODS:
            model, wall = train_source(method, seed, condition, data["train"])
            solve_macs = fit_fusion(model, data["support"])
            payload = Path(outdir) / f"{split_name}_{seed}_{condition}_{method}.npz"
            size, digest = save_payload(model, method, condition, payload)
            loaded = load_model(method, payload_arrays(payload), seed, condition)
            val = evaluate(loaded, data["validation"], condition)
            test = evaluate(loaded, data["test"], condition)
            rows.append({"condition": split_name, "seed": seed, "candidate_family": condition,
                         "method": method, "serialized_bytes": size, "payload_sha256": digest,
                         "optimizer_updates": UPDATES, "source_examples_seen": UPDATES * BATCH,
                         "few_shot_examples_per_target": len(data["support"][0]),
                         "ridge_solve_macs": solve_macs, "train_wall_s": wall,
                         "validation": val, "test": test})
    result = {"condition": split_name, "seed": seed, "target_mix_matrices": target_mixes, "summaries": rows}
    out = Path(json_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=["development", "fresh"], required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    run(args.seed, args.split, args.outdir, args.json)
