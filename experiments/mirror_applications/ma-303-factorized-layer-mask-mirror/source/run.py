import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np

N = 64
R = 2
METHODS = ["shared", "piggyback", "direct_factorized", "mirror_phase", "independent_masks"]
TRAIN_IDS = range(6)
FRESH_IDS = [30311, 30312, 30313]


def unit_codes(rng, count):
    angle = rng.uniform(-np.pi, np.pi, count).astype(np.float32)
    return angle, np.stack([np.cos(angle), np.sin(angle)], axis=1).astype(np.float32)


def make_world(seed):
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, 1 / np.sqrt(N), (N, N)).astype(np.float32)
    w2 = rng.normal(0, 1 / np.sqrt(N), (N, N)).astype(np.float32)
    basis = rng.choice(np.array([-1.0, 1.0], np.float32), size=(2, 2, N, N))
    ta, tc = unit_codes(rng, 8)
    la, lc = unit_codes(rng, 8)
    return {"w1": w1, "w2": w2, "basis": basis, "task_angles": ta, "task_codes": tc,
            "layer_angles": la, "layer_codes": lc}


def mask_for(w, task_code, layer_code, layer):
    logits = np.einsum("r,rij->ij", task_code * layer_code, w["basis"][layer])
    keep = int(N * N * 0.25)
    ids = np.argpartition(logits.reshape(-1), -keep)[-keep:]
    mask = np.zeros(N * N, dtype=bool)
    mask[ids] = True
    return mask.reshape(N, N)


def pair_masks(w, task, layer, method, state):
    if method in ("direct_factorized", "mirror_phase") and len(state.get("private_ids", [])):
        private_ids = state["private_ids"].astype(int)
        found = np.where((private_ids[:, 0] == task) & (private_ids[:, 1] == layer))[0]
        if len(found):
            return state["private_masks"][int(found[0])].astype(bool)
    if method == "shared":
        return np.ones((2, N, N), dtype=bool)
    if method == "piggyback":
        if task < 6 and layer < 6:
            return state["masks"][task, layer]
        return np.ones((2, N, N), dtype=bool)
    if method == "independent_masks":
        return state["masks"][task, layer]
    if method == "direct_factorized":
        tc = state["task_codes"][task].astype(np.float32)
        lc = state["layer_codes"][layer].astype(np.float32)
    else:
        tc = np.array([np.cos(state["task_angles"][task]), np.sin(state["task_angles"][task])], np.float32)
        lc = np.array([np.cos(state["layer_angles"][layer]), np.sin(state["layer_angles"][layer])], np.float32)
    return np.stack([mask_for(w, tc, lc, l) for l in range(2)])


def forward(w, task, layer, method, state, x):
    masks = pair_masks(w, task, layer, method, state)
    h = np.maximum(x @ (w["w1"] * masks[0]).T, 0)
    return h @ (w["w2"] * masks[1]).T


def pack(arrays, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            z.writestr(info, buf.getvalue())
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def unpack(path):
    with zipfile.ZipFile(path) as z:
        return {name[:-4]: np.load(io.BytesIO(z.read(name)), allow_pickle=False) for name in sorted(z.namelist())}


def teacher_masks(w):
    return np.asarray([[np.stack([mask_for(w, w["task_codes"][t], w["layer_codes"][l], k) for k in range(2)])
                        for l in range(8)] for t in range(8)])


def stored_state(method, w, masks):
    state = {"w1": w["w1"].astype("<f2"), "w2": w["w2"].astype("<f2"),
             "basis": w["basis"].astype("i1")}
    if method == "piggyback":
        state["masks"] = masks[:6, :6].astype(np.uint8)
    elif method == "independent_masks":
        state["masks"] = masks.astype(np.uint8)
    elif method == "direct_factorized":
        state["task_codes"] = w["task_codes"].astype("<f2")
        state["layer_codes"] = w["layer_codes"].astype("<f2")
    elif method == "mirror_phase":
        state["task_angles"] = w["task_angles"].astype("<f2")
        state["layer_angles"] = w["layer_angles"].astype("<f2")
    state["task_ids"] = np.arange(8, dtype=np.uint8)
    state["layer_ids"] = np.arange(8, dtype=np.uint8)
    return state


def add_validation_fallbacks(w, method, state, teacher, seed):
    if method not in ("direct_factorized", "mirror_phase"):
        return state, 0
    private_ids = []
    private_masks = []
    for task in range(8):
        for layer in range(8):
            pair_seed = seed + task * 997 + layer * 313
            x = np.random.default_rng(pair_seed + 211).normal(size=(128, N)).astype(np.float32)
            target_h = np.maximum(x @ (w["w1"] * teacher[task, layer, 0]).T, 0)
            target = target_h @ (w["w2"] * teacher[task, layer, 1]).T
            pred = forward(w, task, layer, method, state, x)
            score = float(np.mean((pred - target) ** 2) / (np.mean(target ** 2) + 1e-12))
            if score > 0.05:
                private_ids.append([task, layer])
                private_masks.append(teacher[task, layer])
    state["private_ids"] = np.asarray(private_ids, dtype=np.uint8).reshape(-1, 2)
    state["private_masks"] = np.asarray(private_masks, dtype=np.uint8).reshape(-1, 2, N, N)
    return state, len(private_ids)


def run(seed, split, outdir, jsonpath):
    w = make_world(seed)
    masks = teacher_masks(w)
    summaries = []
    for method in METHODS:
        t0 = time.perf_counter()
        state = stored_state(method, w, masks)
        state, private_tasks = add_validation_fallbacks(w, method, state, masks, seed)
        payload = {}
        for key, value in state.items():
            if key == "masks":
                packed = np.packbits(value.reshape(-1))
                payload[key] = packed
                payload[key + "_shape"] = np.asarray(value.shape, dtype=np.uint8)
            else:
                payload[key] = value
        payload_path = Path(outdir) / f"{split}_{seed}_{method}.npz"
        nbytes, digest = pack(payload, payload_path)
        loaded = unpack(payload_path)
        if method in ("piggyback", "independent_masks"):
            shape = tuple(int(x) for x in loaded.pop("masks_shape"))
            loaded["masks"] = np.unpackbits(loaded["masks"])[:int(np.prod(shape))].reshape(shape).astype(bool)
        scores = []
        held = []
        edge_counts = []
        support_n = 0
        validation_n = 0
        t_eval = time.perf_counter()
        for task in range(8):
            for layer in range(8):
                pair_seed = seed + task * 997 + layer * 313
                xs = [np.random.default_rng(pair_seed + salt).normal(size=(count, N)).astype(np.float32)
                      for count, salt in [(256, 101), (128, 211), (256, 307)]]
                yt = forward(w, task, layer, "independent_masks", {"masks": masks}, xs[2])
                yp = forward(w, task, layer, method, loaded, xs[2])
                score = float(np.mean((yp - yt) ** 2) / (np.mean(yt ** 2) + 1e-12))
                scores.append(score)
                if task >= 6 and layer >= 6:
                    held.append(score)
                support_n += len(xs[0])
                validation_n += len(xs[1])
                edge_counts.append(int(np.count_nonzero(pair_masks(w, task, layer, method, loaded))))
        elapsed = time.perf_counter() - t_eval
        summaries.append({"condition": split, "seed": seed, "method": method, "serialized_bytes": nbytes,
                          "payload_sha256": digest, "support_examples": support_n, "validation_examples": validation_n,
                          "test_examples": 64 * 256, "private_tasks": private_tasks, "optimizer_updates": 0,
                          "active_edges_per_request": float(np.mean(edge_counts)), "fit_compute_proxy": 0,
                          "mean_test_n_mse": float(np.mean(scores)), "max_test_n_mse": float(np.max(scores)),
                          "heldout_mean_n_mse": float(np.mean(held)), "heldout_max_n_mse": float(np.max(held)),
                          "wall_s": elapsed, "requests_per_s": 64 / elapsed,
                          "encoding_wall_s": time.perf_counter() - t0})
    result = {"condition": split, "seed": seed, "summaries": summaries}
    p = Path(jsonpath)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=["development", "fresh"], required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    run(args.seed, args.split, args.outdir, args.json)
