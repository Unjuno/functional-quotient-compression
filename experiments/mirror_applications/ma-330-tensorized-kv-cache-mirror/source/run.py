import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch


torch.set_num_threads(1)
LAYERS, HEADS, TOKENS, DIM = 4, 4, 128, 16
ALIGNED_LAYERS = 3
TEMPORAL_RANK = 4
METHODS = ["independent_full", "mlkv_hard_share", "shared_direct", "mirror_phase", "mirror_no_private"]


def rotate(x, phase):
    c, s = torch.cos(phase), torch.sin(phase)
    pairs = x.reshape(*x.shape[:-1], DIM // 2, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    return torch.stack((c * a - s * b, s * a + c * b), dim=-1).reshape_as(x)


def world(seed):
    g = torch.Generator().manual_seed(seed)
    time_basis = torch.linalg.qr(torch.randn(TOKENS, TEMPORAL_RANK, generator=g), mode="reduced").Q
    k_coeff = torch.randn(HEADS, TEMPORAL_RANK, DIM, generator=g) * 0.35
    v_coeff = torch.randn(HEADS, TEMPORAL_RANK, DIM, generator=g) * 0.35
    base_k = torch.einsum("tr,hrd->htd", time_basis, k_coeff)
    base_v = torch.einsum("tr,hrd->htd", time_basis, v_coeff)
    phases = (torch.rand(LAYERS, HEADS, generator=g) - 0.5) * 1.4
    target_k = torch.empty(LAYERS, HEADS, TOKENS, DIM)
    target_v = torch.empty_like(target_k)
    queries = torch.empty_like(target_k)
    unrelated_k = torch.randn(HEADS, TOKENS, DIM, generator=g) * 0.35
    unrelated_v = torch.randn(HEADS, TOKENS, DIM, generator=g) * 0.35
    for l in range(LAYERS):
        for h in range(HEADS):
            q = torch.randn(TOKENS, DIM, generator=g) * 0.5
            if l < ALIGNED_LAYERS:
                target_k[l, h] = rotate(base_k[h], phases[l, h])
                target_v[l, h] = rotate(base_v[h], phases[l, h])
                queries[l, h] = rotate(q, phases[l, h])
            else:
                target_k[l, h] = unrelated_k[h]
                target_v[l, h] = unrelated_v[h]
                queries[l, h] = q
    return base_k, base_v, phases, target_k, target_v, queries, unrelated_k, unrelated_v, time_basis, k_coeff, v_coeff


def attention(q, k, v):
    # Causal attention over each prefix; output at all positions.
    outputs = []
    for t in range(TOKENS):
        score = q[t:t+1] @ k[:t+1].T / (DIM ** 0.5)
        outputs.append(torch.softmax(score, dim=-1) @ v[:t+1])
    return torch.cat(outputs, dim=0)


def reference_outputs(target_k, target_v, queries):
    out = torch.empty_like(target_k)
    with torch.no_grad():
        for l in range(LAYERS):
            for h in range(HEADS):
                out[l, h] = attention(queries[l, h], target_k[l, h], target_v[l, h])
    return out


def deterministic_pack(arrays, path):
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


def payload(method, world_data, path):
    base_k, base_v, phases, target_k, target_v, _, _, _, time_basis, k_coeff, v_coeff = world_data
    f16 = lambda x: x.detach().cpu().numpy().astype("<f2")
    arr = {"meta": np.asarray([LAYERS, HEADS, TOKENS, DIM, ALIGNED_LAYERS], dtype=np.uint16)}
    if method == "independent_full":
        arr.update(k=f16(target_k), v=f16(target_v))
    elif method == "mlkv_hard_share":
        arr.update(time_basis=f16(time_basis), k_coeff=f16(k_coeff), v_coeff=f16(v_coeff))
    else:
        arr.update(time_basis=f16(time_basis), k_coeff=f16(k_coeff), v_coeff=f16(v_coeff))
        if method == "shared_direct":
            arr["rotation"] = f16(torch.stack([torch.cos(phases[:ALIGNED_LAYERS]), torch.sin(phases[:ALIGNED_LAYERS])], dim=-1))
        elif method == "mirror_phase":
            arr["phase"] = f16(phases[:ALIGNED_LAYERS])
        elif method == "mirror_no_private":
            arr["phase"] = f16(phases)
        if method in ("shared_direct", "mirror_phase"):
            arr["private_k"] = f16(target_k[ALIGNED_LAYERS])
            arr["private_v"] = f16(target_v[ALIGNED_LAYERS])
    n, digest = deterministic_pack(arr, path)
    return n, digest, arr


def load_arrays(path):
    arrays = {}
    with zipfile.ZipFile(path) as z:
        for member in z.namelist():
            arrays[member[:-4]] = np.load(io.BytesIO(z.read(member)), allow_pickle=False)
    return arrays


def unpack_cache(method, arrays):
    if method == "independent_full":
        k = torch.from_numpy(np.array(arrays["k"], copy=True)).float()
        v = torch.from_numpy(np.array(arrays["v"], copy=True)).float()
        return k, v
    basis = torch.from_numpy(np.array(arrays["time_basis"], copy=True)).float()
    k_coeff = torch.from_numpy(np.array(arrays["k_coeff"], copy=True)).float()
    v_coeff = torch.from_numpy(np.array(arrays["v_coeff"], copy=True)).float()
    k = torch.einsum("tr,hrd->htd", basis, k_coeff)
    v = torch.einsum("tr,hrd->htd", basis, v_coeff)
    if method == "mlkv_hard_share":
        return k.unsqueeze(0).expand(LAYERS, -1, -1, -1), v.unsqueeze(0).expand(LAYERS, -1, -1, -1)
    if "phase" in arrays:
        phase = torch.from_numpy(np.array(arrays["phase"], copy=True)).float()
        co, si = torch.cos(phase), torch.sin(phase)
    else:
        cs = torch.from_numpy(np.array(arrays["rotation"], copy=True)).float()
        co, si = cs[..., 0], cs[..., 1]
    out_k, out_v = torch.empty(LAYERS, HEADS, TOKENS, DIM), torch.empty(LAYERS, HEADS, TOKENS, DIM)
    logical_layers = LAYERS if method == "mirror_no_private" else ALIGNED_LAYERS
    for l in range(logical_layers):
        for h in range(HEADS):
            phase = torch.atan2(si[l, h], co[l, h]) if "rotation" in arrays else torch.atan2(si[l, h], co[l, h])
            out_k[l, h] = rotate(k[h], phase)
            out_v[l, h] = rotate(v[h], phase)
    if method != "mirror_no_private":
        out_k[ALIGNED_LAYERS] = torch.from_numpy(np.array(arrays["private_k"], copy=True)).float()
        out_v[ALIGNED_LAYERS] = torch.from_numpy(np.array(arrays["private_v"], copy=True)).float()
    return out_k, out_v


def evaluate(method, arrays, target_output, queries):
    recon_start = time.perf_counter()
    k, v = unpack_cache(method, arrays)
    reconstruction_wall = time.perf_counter() - recon_start
    out = torch.empty_like(target_output)
    start = time.perf_counter()
    with torch.no_grad():
        for l in range(LAYERS):
            for h in range(HEADS):
                out[l, h] = attention(queries[l, h], k[l, h], v[l, h])
    attention_wall = time.perf_counter() - start
    wall = reconstruction_wall + attention_wall
    sqerr = (out - target_output) ** 2
    mse = float(sqerr.mean())
    nmse = mse / float((target_output ** 2).mean())
    layer_nmse = []
    for l in range(LAYERS):
        layer_nmse.append(float(sqerr[l].mean() / (target_output[l] ** 2).mean()))
    return {"output_mse": mse, "output_nmse": nmse, "decode_wall_s": wall,
            "reconstruction_wall_s": reconstruction_wall, "attention_wall_s": attention_wall,
            "queries_per_s": LAYERS * HEADS * TOKENS / wall, "layer_output_nmse": layer_nmse}


def run(seed, split, outdir, jsonpath):
    data = world(seed)
    target = reference_outputs(data[3], data[4], data[5])
    rows = []
    for method in METHODS:
        path = Path(outdir) / f"{split}_{seed}_{method}.zip"
        n, digest, _ = payload(method, data, path)
        arr = load_arrays(path)
        metrics = evaluate(method, arr, target, data[5])
        # Multiply-add proxy for causal attention plus K/V reconstruction.
        attention_macs = 2 * LAYERS * HEADS * TOKENS * TOKENS * DIM
        base_decode_macs = 2 * HEADS * TOKENS * TEMPORAL_RANK * DIM
        view_layers = LAYERS if method == "mirror_no_private" else ALIGNED_LAYERS
        view_macs = 2 * view_layers * HEADS * TOKENS * DIM * 2
        reconstruction_macs = 0 if method == "independent_full" else base_decode_macs
        if method not in ("independent_full", "mlkv_hard_share"):
            reconstruction_macs += view_macs
        rows.append({"condition": split, "seed": seed, "method": method, "serialized_bytes": n,
                     "payload_sha256": digest, "active_attention_macs_proxy": attention_macs,
                     "reconstruction_macs_proxy": reconstruction_macs, "updates": 0,
                     "examples_seen": LAYERS * HEADS * TOKENS,
                     "output_mse": metrics["output_mse"], "output_nmse": metrics["output_nmse"],
                     "layer_output_nmse": metrics["layer_output_nmse"],
                     "decode_wall_s": metrics["decode_wall_s"],
                     "reconstruction_wall_s": metrics["reconstruction_wall_s"],
                     "attention_wall_s": metrics["attention_wall_s"],
                     "queries_per_s": metrics["queries_per_s"]})
    result = {"condition": split, "seed": seed, "summaries": rows}
    Path(jsonpath).parent.mkdir(parents=True, exist_ok=True)
    Path(jsonpath).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", choices=["development", "fresh"], required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    run(args.seed, args.split, args.outdir, args.json)
