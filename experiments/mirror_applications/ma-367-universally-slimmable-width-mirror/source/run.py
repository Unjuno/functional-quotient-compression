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

from model import INPUT_DIM, NUM_CLASSES, WIDTHS, METHODS, IndependentWidths, SharedSlimmableMLP

torch.set_num_threads(1)
UPDATES = 1000
BATCH = 128
LR = 1e-3


def teacher(seed):
    gen = torch.Generator().manual_seed(seed + 9000)
    w1 = torch.randn(INPUT_DIM, max(WIDTHS), generator=gen) * 0.28
    b1 = torch.randn(max(WIDTHS), generator=gen) * 0.08
    w2 = torch.randn(max(WIDTHS), NUM_CLASSES, generator=gen) * 0.35
    b2 = torch.randn(NUM_CLASSES, generator=gen) * 0.08
    return w1, b1, w2, b2


def make_data(seed):
    w1, b1, w2, b2 = teacher(seed)
    splits = {}
    for name, n, salt in (("train", 8192, 101), ("validation", 2048, 211), ("test", 4096, 307)):
        gen = torch.Generator().manual_seed(seed + salt)
        x = torch.randn(n, INPUT_DIM, generator=gen)
        with torch.no_grad():
            logits = torch.relu(x @ w1 + b1) @ w2 + b2
            y = logits.argmax(dim=-1)
        splits[name] = (x, y)
    return splits


def sampled_widths(seed, step):
    rng = np.random.default_rng(seed * 100003 + step)
    mid = int(rng.choice([WIDTHS[1], WIDTHS[2]]))
    return tuple(sorted({WIDTHS[0], WIDTHS[-1], mid}))


def train(method, seed, splits):
    x, y = splits["train"]
    if method == "independent":
        model = IndependentWidths(seed * 37 + 5)
    else:
        model = SharedSlimmableMLP(method, seed * 37 + METHODS.index(method))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    start = time.perf_counter()
    last_batch_widths = None
    for step in range(UPDATES):
        ids_rng = np.random.default_rng(seed * 100003 + step + 55001)
        ids = torch.from_numpy(ids_rng.integers(0, len(x), size=BATCH, dtype=np.int64))
        active_widths = sampled_widths(seed, step)
        last_batch_widths = active_widths
        losses = []
        for width in active_widths:
            logits = model.forward_width(width, x[ids])
            losses.append(F.cross_entropy(logits, y[ids]))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval(), time.perf_counter() - start


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


def save_payload(model, method, path):
    arrays = {}
    for k, value in model.state_dict().items():
        a = value.detach().cpu().numpy()
        arrays[k] = a.astype("<f2") if a.dtype.kind == "f" else a
    arrays["widths"] = np.asarray(WIDTHS, dtype=np.uint8)
    arrays["meta"] = np.asarray([INPUT_DIM, NUM_CLASSES, max(WIDTHS), len(WIDTHS)], dtype=np.uint16)
    arrays["method_id"] = np.asarray([METHODS.index(method)], dtype=np.uint8)
    n, digest = deterministic_pack(arrays, path)
    return n, digest, arrays


def payload_arrays(path):
    arrays = {}
    with zipfile.ZipFile(path, "r") as archive:
        for member in archive.namelist():
            arrays[member[:-4]] = np.lib.format.read_array(io.BytesIO(archive.read(member)), allow_pickle=False)
    return arrays


def reload_model(method, arrays, seed):
    if method == "independent":
        model = IndependentWidths(seed * 37 + 5)
    else:
        model = SharedSlimmableMLP(method, seed * 37 + METHODS.index(method))
    ref = model.state_dict()
    state = {}
    for k, value in ref.items():
        arr = torch.from_numpy(np.array(arrays[k], copy=True))
        state[k] = arr.to(value.dtype) if value.is_floating_point() else arr
    model.load_state_dict(state)
    return model.eval()


def evaluate(model, method, split):
    x, y = split
    nll, acc, throughput, macs = [], [], [], []
    for width in WIDTHS:
        with torch.no_grad():
            start = time.perf_counter()
            chunks = []
            for j in range(0, len(x), 512):
                chunks.append(model.forward_width(width, x[j:j + 512]))
            elapsed = time.perf_counter() - start
            logits = torch.cat(chunks)
            nll.append(float(F.cross_entropy(logits, y)))
            acc.append(float((logits.argmax(-1) == y).float().mean()))
            throughput.append(float(len(x) / max(elapsed, 1e-9)))
        base = INPUT_DIM * width + width * NUM_CLASSES
        # Each adjacent pair uses four multiplications and two additions,
        # counted as two multiply-accumulate equivalents per pair.
        view = (2 * width) if method == "mirror_givens" else 0
        macs.append(float(base + view))
    return {"width_nll": nll, "width_accuracy": acc, "width_examples_per_s": throughput,
            "width_active_macs_per_example": macs, "macro_nll": float(np.mean(nll)),
            "macro_accuracy": float(np.mean(acc))}


def run(seed, split_name, outdir, json_path):
    splits = make_data(seed)
    rows = []
    for method in METHODS:
        model, wall = train(method, seed, splits)
        payload = Path(outdir) / f"{split_name}_{seed}_{method}.npz"
        size, digest, _ = save_payload(model, method, payload)
        arrays = payload_arrays(payload)
        loaded = reload_model(method, arrays, seed)
        val = evaluate(loaded, method, splits["validation"])
        test = evaluate(loaded, method, splits["test"])
        rows.append({"condition": split_name, "seed": seed, "method": method,
                     "serialized_bytes": size, "payload_sha256": digest,
                     "optimizer_updates": UPDATES, "train_examples_per_width": UPDATES * BATCH,
                     "active_widths_per_update": 3, "train_wall_s": wall,
                     "validation_width_nll": val["width_nll"], "validation_width_accuracy": val["width_accuracy"],
                     "test_width_nll": test["width_nll"], "test_width_accuracy": test["width_accuracy"],
                     "width_active_macs_per_example": test["width_active_macs_per_example"],
                     "width_examples_per_s": test["width_examples_per_s"],
                     "macro_validation_nll": val["macro_nll"], "macro_test_nll": test["macro_nll"],
                     "macro_test_accuracy": test["macro_accuracy"]})
    result = {"condition": split_name, "seed": seed, "widths": list(WIDTHS), "summaries": rows}
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
