from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np


WIDTHS = (4, 8, 12, 16, 20, 24, 28, 32)
TRAIN_WIDTHS = (8, 16, 24, 32)
HELDOUT_WIDTHS = (4, 12, 20, 28)
INPUT_DIM = OUTPUT_DIM = 16
HIDDEN_DIM = 32
N_TRAIN = N_HELDOUT = 512


def pair_rotate(h: np.ndarray, theta: float) -> np.ndarray:
    out = h.copy()
    c, s = np.cos(theta), np.sin(theta)
    even = h[..., 0::2]
    odd = h[..., 1::2]
    out[..., 0::2] = c * even - s * odd
    out[..., 1::2] = s * even + c * odd
    return out


def maps(U: np.ndarray, V: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the cosine and sine output maps for an active k-channel prefix."""
    Uk, Vk = U[:, :k], V[:k, :]
    pairs = k // 2
    p = np.zeros((INPUT_DIM, OUTPUT_DIM), dtype=np.float64)
    q = np.zeros_like(p)
    for j in range(pairs):
        u0, u1 = Uk[:, 2 * j], Uk[:, 2 * j + 1]
        v0, v1 = Vk[2 * j], Vk[2 * j + 1]
        p += np.outer(u0, v0) + np.outer(u1, v1)
        q += np.outer(u0, v1) - np.outer(u1, v0)
    return p, q


def output_map(U: np.ndarray, V: np.ndarray, k: int, theta: float) -> np.ndarray:
    p, q = maps(U, V, k)
    return np.cos(theta) * p + np.sin(theta) * q


def teacher_world(world: int) -> dict:
    rng = np.random.default_rng(world)
    U = rng.normal(0, 1 / np.sqrt(INPUT_DIM), (INPUT_DIM, HIDDEN_DIM)).astype(np.float32)
    V = rng.normal(0, 1 / np.sqrt(HIDDEN_DIM), (HIDDEN_DIM, OUTPUT_DIM)).astype(np.float32)
    a = rng.uniform(-0.20, 0.20)
    b = rng.uniform(0.25, 0.55)
    data = {}
    for k in WIDTHS:
        qwidth = k / HIDDEN_DIM
        theta = a + b * qwidth
        xt = rng.normal(size=(N_TRAIN, INPUT_DIM))
        xv = rng.normal(size=(N_HELDOUT, INPUT_DIM))
        A = output_map(U, V, k, theta)
        data[k] = (xt, xt @ A, xv, xv @ A)
    return {"U": U, "V": V, "a_true": a, "b_true": b, "data": data}


def estimate_width_angle(U: np.ndarray, V: np.ndarray, k: int, x: np.ndarray, y: np.ndarray) -> float:
    p, q = maps(U, V, k)
    xp, xq = x @ p, x @ q
    design = np.stack([xp.reshape(-1), xq.reshape(-1)], axis=1)
    cs, *_ = np.linalg.lstsq(design, y.reshape(-1), rcond=None)
    cs /= np.linalg.norm(cs)
    return float(np.arctan2(cs[1], cs[0]))


def fit_codes(world: dict) -> tuple[np.ndarray, np.ndarray]:
    U, V, data = world["U"], world["V"], world["data"]
    qs, angles = [], []
    design_rows, target_rows = [], []
    for k in TRAIN_WIDTHS:
        x, y, _, _ = data[k]
        qwidth = k / HIDDEN_DIM
        qs.append(qwidth)
        angles.append(estimate_width_angle(U, V, k, x, y))
        p, s = maps(U, V, k)
        xp, xs = x @ p, x @ s
        # Generic direct coefficient control: c(q)=c0+c1q and s(q)=s0+s1q.
        design_rows.append(np.stack([xp, qwidth * xp, xs, qwidth * xs], axis=-1).reshape(-1, 4))
        target_rows.append(y.reshape(-1))
    angle_fit = np.polyfit(np.asarray(qs), np.asarray(angles), deg=1)
    coef, *_ = np.linalg.lstsq(np.concatenate(design_rows), np.concatenate(target_rows), rcond=None)
    return angle_fit.astype(np.float32), coef.astype(np.float32)


def nrmse(pred: np.ndarray, target: np.ndarray) -> float:
    denom = np.mean(target * target)
    return float(np.sqrt(np.mean((pred - target) ** 2) / max(denom, 1e-30)))


def predict(U, V, k, method, angle_fit, coeff):
    qwidth = k / HIDDEN_DIM
    if method == "usnet_prefix":
        return output_map(U, V, k, 0.0), 0
    if method == "mirror_angle":
        theta = angle_fit[0] * qwidth + angle_fit[1]
        return output_map(U, V, k, float(theta)), 2 * (k // 2)
    if method == "direct_coeff":
        c0, c1, s0, s1 = coeff
        p, s = maps(U, V, k)
        return (c0 + c1 * qwidth) * p + (s0 + s1 * qwidth) * s, 4 * INPUT_DIM * OUTPUT_DIM
    raise ValueError(method)


def payload(path: Path, method: str, U, V, angle_fit, coeff, independent_maps=None):
    if method == "usnet_prefix":
        np.savez(path, U=U.astype(np.float32), V=V.astype(np.float32), widths=np.asarray(WIDTHS, np.int16), method=np.asarray(method))
    elif method == "mirror_angle":
        np.savez(path, U=U.astype(np.float32), V=V.astype(np.float32), angle_fit=angle_fit, widths=np.asarray(WIDTHS, np.int16), method=np.asarray(method))
    elif method == "direct_coeff":
        np.savez(path, U=U.astype(np.float32), V=V.astype(np.float32), coefficients=coeff, widths=np.asarray(WIDTHS, np.int16), method=np.asarray(method))
    elif method == "independent_full":
        np.savez(path, operators=np.stack(independent_maps).astype(np.float32), widths=np.asarray(WIDTHS, np.int16), method=np.asarray(method))
    else:
        raise ValueError(method)
    return path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()


def run_world(world_id: int, split: str, outdir: Path) -> list[dict]:
    world = teacher_world(world_id)
    U, V, data = world["U"], world["V"], world["data"]
    angle_fit, coeff = fit_codes(world)
    outdir.mkdir(parents=True, exist_ok=True)
    methods = ("usnet_prefix", "mirror_angle", "direct_coeff", "independent_full")
    records = []
    for method in methods:
        path = outdir / f"{method}.npz"
        ind_maps = [output_map(U, V, k, world["a_true"] + world["b_true"] * k / HIDDEN_DIM).astype(np.float32).astype(np.float64) for k in WIDTHS]
        nbytes, digest = payload(path, method, U, V, angle_fit, coeff, ind_maps)
        for k in WIDTHS:
            _, _, x, y = data[k]
            decode_start = time.perf_counter()
            if method == "independent_full":
                A = ind_maps[WIDTHS.index(k)]
                macs = 2 * INPUT_DIM * OUTPUT_DIM
            else:
                A, extra = predict(U, V, k, method, angle_fit, coeff)
                macs = 2 * INPUT_DIM * k + 2 * k * OUTPUT_DIM + extra
            decode_seconds = time.perf_counter() - decode_start
            start = time.perf_counter()
            pred = None
            for _ in range(20):
                pred = x @ A
            forward_seconds = (time.perf_counter() - start) / 20
            assert pred is not None
            records.append({
                "world": world_id, "split": split, "method": method, "width": k,
                "heldout_examples": int(len(x)), "optimizer_updates": 0,
                "payload_bytes": nbytes, "payload_sha256": digest,
                "heldout_nrmse": nrmse(pred, y), "macs_per_example_proxy": int(macs),
                "decode_wall_seconds": decode_seconds,
                "forward_wall_seconds_per_batch": forward_seconds,
                "wall_seconds": decode_seconds + forward_seconds,
                "theta_slope": float(angle_fit[0]) if method == "mirror_angle" else "",
                "theta_intercept": float(angle_fit[1]) if method == "mirror_angle" else ""
            })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--split", choices=("dev", "fresh"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = run_world(args.world, args.split, args.out)
    out = args.out / "RESULTS_CORE.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"world": args.world, "split": args.split, "rows": len(rows), "path": str(out)}))


if __name__ == "__main__":
    main()
