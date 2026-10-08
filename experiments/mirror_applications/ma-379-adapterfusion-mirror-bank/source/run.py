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

from model import ADAPTER_DIM, INPUT_DIM, METHODS, SOURCE_TASKS, TARGET_TASKS, AdapterBank

torch.set_num_threads(1)
UPDATES, BATCH, LR = 1600, 64, 1e-3


def teacher(seed):
    g = torch.Generator().manual_seed(seed + 39000)
    sources = torch.randn(SOURCE_TASKS, INPUT_DIM, ADAPTER_DIM, generator=g) * 0.06
    mix = torch.eye(TARGET_TASKS, SOURCE_TASKS) * 0.58
    for t in range(TARGET_TASKS):
        mix[t, (t + 1) % SOURCE_TASKS] = 0.27
        mix[t, (t + 2) % SOURCE_TASKS] = 0.15
    targets = torch.einsum("tk,kdf->tdf", mix, sources)
    return sources, targets, mix


def make_data(seed):
    sources, targets, mix = teacher(seed)
    data = {}
    for name, n, salt in (("train", 8192, 101), ("validation", 2048, 211), ("test", 4096, 307)):
        g = torch.Generator().manual_seed(seed + salt)
        x = torch.randn(n, INPUT_DIM, generator=g)
        src = torch.einsum("bd,kdf->bkf", x, sources)
        tgt = torch.einsum("bd,tdf->btf", x, targets)
        data[name] = (x, src, tgt)
    return data, mix


def train(method, seed, data):
    torch.manual_seed(seed * 31 + METHODS.index(method))
    model = AdapterBank(method, seed * 67 + METHODS.index(method))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    x, source_targets, target_targets = data["train"]
    rng = np.random.default_rng(seed * 100003 + 29011)
    start = time.perf_counter()
    for _ in range(UPDATES):
        ids = torch.from_numpy(rng.integers(0, len(x), size=BATCH, dtype=np.int64))
        source_pred = model.source_outputs(x[ids])
        target_pred = model.target_outputs(x[ids])
        source_loss = F.mse_loss(source_pred, source_targets[ids])
        target_loss = F.mse_loss(target_pred, target_targets[ids])
        loss = source_loss + target_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval(), time.perf_counter() - start


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


def save_payload(model, method, path):
    arrays = {}
    for key, tensor in model.state_dict().items():
        value = tensor.detach().cpu().numpy()
        arrays[key] = value.astype("<f2") if value.dtype.kind == "f" else value
    arrays["meta"] = np.asarray([INPUT_DIM, ADAPTER_DIM, SOURCE_TASKS, TARGET_TASKS], dtype=np.uint16)
    arrays["method_id"] = np.asarray([METHODS.index(method)], dtype=np.uint8)
    arrays["source_ids"] = np.arange(SOURCE_TASKS, dtype=np.uint8)
    arrays["target_ids"] = np.arange(TARGET_TASKS, dtype=np.uint8)
    return deterministic_pack(arrays, path)


def load_model(method, arrays, seed):
    model = AdapterBank(method, seed * 67 + METHODS.index(method))
    state = {}
    for key, value in model.state_dict().items():
        tensor = torch.from_numpy(np.array(arrays[key], copy=True))
        state[key] = tensor.to(value.dtype) if value.is_floating_point() else tensor
    model.load_state_dict(state)
    return model.eval()


def evaluate(model, split):
    x, source_targets, target_targets = split
    with torch.no_grad():
        source_pred = model.source_outputs(x)
        source_mses = ((source_pred - source_targets) ** 2).mean(dim=(0, 2)).tolist()
        start = time.perf_counter()
        target_pred = model.target_outputs(x)
        elapsed = time.perf_counter() - start
        target_mses = ((target_pred - target_targets) ** 2).mean(dim=(0, 2)).tolist()
    targets_per_s = float(len(x) * TARGET_TASKS / max(elapsed, 1e-9))
    independent_mac = SOURCE_TASKS * INPUT_DIM * ADAPTER_DIM + SOURCE_TASKS * ADAPTER_DIM
    view_mac = 0
    if model.method == "mirror_shared":
        view_mac = SOURCE_TASKS * 3 * ADAPTER_DIM ** 3
    return {"source_mses": source_mses, "target_mses": target_mses,
            "mean_source_mse": float(np.mean(source_mses)), "mean_target_mse": float(np.mean(target_mses)),
            "target_examples_per_s": targets_per_s,
            "macs_per_target_example": independent_mac + view_mac,
            "fusion_coefficients": model.fusion.detach().cpu().tolist(),
            "source_matrix_norms": model.source_matrices().detach().norm(dim=(1, 2)).cpu().tolist()}


def run(seed, split_name, outdir, json_path):
    data, mix = make_data(seed)
    rows = []
    for method in METHODS:
        model, wall = train(method, seed, data)
        payload = Path(outdir) / f"{split_name}_{seed}_{method}.npz"
        size, digest = save_payload(model, method, payload)
        loaded = load_model(method, payload_arrays(payload), seed)
        validation = evaluate(loaded, data["validation"])
        test = evaluate(loaded, data["test"])
        rows.append({"condition": split_name, "seed": seed, "method": method,
                     "serialized_bytes": size, "payload_sha256": digest,
                     "optimizer_updates": UPDATES, "examples_seen": UPDATES * BATCH,
                     "train_wall_s": wall, "validation": validation, "test": test})
    result = {"condition": split_name, "seed": seed, "target_mix_matrix": mix.tolist(), "summaries": rows}
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=["development", "fresh"], required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    run(args.seed, args.split, args.outdir, args.json)
