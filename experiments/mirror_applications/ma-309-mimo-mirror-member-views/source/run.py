import argparse
import hashlib
import io
import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

M = 4
torch.set_num_threads(1)
IN = 2
H = 32
CLASSES = 2
UPDATES = 500
BATCH = 64
LR = 0.01
METHODS = ["mimo_heads", "hard_tied", "scalar_gate", "mirror_givens", "independent_mlps"]
RULE_NAMES = ["sign_x0", "sign_x1", "xor_signs", "outside_radius"]


def rule_labels(x, member):
    if member == 0:
        return (x[:, 0] > 0).long()
    if member == 1:
        return (x[:, 1] > 0).long()
    if member == 2:
        return ((x[:, 0] * x[:, 1]) > 0).long()
    return ((x[:, 0] ** 2 + x[:, 1] ** 2) > 2.5).long()


def data_world(seed):
    rng = np.random.default_rng(seed)
    splits = {}
    for split, n, offset in [("train", 32000, 100), ("validation", 1024, 10000), ("test", 2048, 20000)]:
        xs, ys = [], []
        for member in range(M):
            x = rng.normal(size=(n, IN)).astype(np.float32)
            # Keep each task stream independent and reproducible by seed/member.
            x = np.random.default_rng(seed + offset + member * 131).normal(size=(n, IN)).astype(np.float32)
            xs.append(x)
            ys.append(rule_labels(torch.from_numpy(x), member).numpy().astype(np.int64))
        splits[split] = (np.stack(xs), np.stack(ys))
    common = np.random.default_rng(seed + 77777).normal(size=(4096, IN)).astype(np.float32)
    return splits, common


class Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(IN, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())

    def forward(self, x):
        return self.layers(x)


class MimoModel(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        self.trunks = nn.ModuleList([Trunk() for _ in range(M)]) if method == "independent_mlps" else nn.ModuleList([Trunk()])
        if method in ("mimo_heads", "independent_mlps"):
            self.heads = nn.ModuleList([nn.Linear(H, CLASSES) for _ in range(M)])
        else:
            self.head = nn.Linear(H, CLASSES)
        if method == "scalar_gate":
            self.scales = nn.Parameter(torch.ones(M))
        if method == "mirror_givens":
            self.angles = nn.Parameter(torch.zeros(M))

    def forward(self, xs):
        # xs has shape [member, batch, input], so all member streams share one MIMO call.
        outputs = []
        for member in range(M):
            trunk_idx = member if self.method == "independent_mlps" else 0
            h = self.trunks[trunk_idx](xs[member])
            if self.method == "mirror_givens":
                a = self.angles[member]
                c, s = torch.cos(a), torch.sin(a)
                u, v = h[:, 0], h[:, 1]
                h = h.clone()
                h[:, 0] = c * u - s * v
                h[:, 1] = s * u + c * v
            if self.method in ("mimo_heads", "independent_mlps"):
                y = self.heads[member](h)
            else:
                y = self.head(h)
            if self.method == "scalar_gate":
                y = self.scales[member] * y
            outputs.append(y)
        return torch.stack(outputs)


def train(method, seed, splits):
    torch.manual_seed(seed)
    model = MimoModel(method)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    xtrain, ytrain = splits["train"]
    xtrain = torch.from_numpy(xtrain)
    ytrain = torch.from_numpy(ytrain)
    rng = np.random.default_rng(seed + 999)
    model.train()
    start = time.perf_counter()
    loss_total = 0.0
    for step in range(UPDATES):
        ids = rng.integers(0, len(xtrain[0]), size=BATCH)
        xb = xtrain[:, ids]
        yb = ytrain[:, ids]
        logits = model(xb)
        loss = sum(F.cross_entropy(logits[m], yb[m]) for m in range(M)) / M
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_total += float(loss.detach())
    wall = time.perf_counter() - start
    return model.eval(), wall, loss_total / UPDATES


def ece(logits, labels, bins=10):
    probs = torch.softmax(logits, dim=-1)
    conf, pred = probs.max(dim=-1)
    acc = (pred == labels).float()
    edges = torch.linspace(0, 1, bins + 1)
    value = torch.tensor(0.0)
    for i in range(bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1] if i < bins - 1 else conf <= edges[i + 1])
        if mask.any():
            value += mask.float().mean() * torch.abs(acc[mask].mean() - conf[mask].mean())
    return float(value)


def payload(model, path):
    arrays = {}
    for key, tensor in model.state_dict().items():
        arr = tensor.detach().cpu().numpy()
        if arr.dtype.kind == "f":
            arr = arr.astype("<f2")
        arrays[key] = arr
    arrays["member_ids"] = np.arange(M, dtype=np.uint8)
    arrays["meta"] = np.asarray([IN, H, CLASSES, M, UPDATES], dtype=np.uint16)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_STORED) as z:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            z.writestr(info, buf.getvalue())
    raw = p.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest(), arrays


def diversity(logits):
    # logits: [member, common_examples, classes]
    pred = logits.argmax(dim=-1).cpu().numpy()
    vals = [float(np.mean(pred[i] != pred[j])) for i in range(M) for j in range(i + 1, M)]
    return float(np.mean(vals)), vals


def evaluate(model, splits, common):
    model.eval()
    xte, yte = splits["test"]
    with torch.no_grad():
        logits = model(torch.from_numpy(xte))
        y = torch.from_numpy(yte)
        acc, losses, eces = [], [], []
        for member in range(M):
            acc.append(float((logits[member].argmax(-1) == y[member]).float().mean()))
            losses.append(float(F.cross_entropy(logits[member], y[member])))
            eces.append(ece(logits[member], y[member]))
        common_batch = np.repeat(common[None, :, :], M, axis=0)
        common_logits = model(torch.from_numpy(common_batch))
        div, pairs = diversity(common_logits)
        start = time.perf_counter()
        for _ in range(20):
            model(torch.from_numpy(xte))
        elapsed = (time.perf_counter() - start) / 20
    return {"member_accuracy": acc, "member_nll": losses, "member_ece": eces,
            "macro_accuracy": float(np.mean(acc)), "macro_nll": float(np.mean(losses)),
            "macro_ece": float(np.mean(eces)), "pairwise_disagreement": div,
            "pairwise_disagreement_values": pairs, "inference_wall_s_per_call": elapsed,
            "examples_per_s": float(M * len(xte[0]) / elapsed)}


def run(seed, split, outdir, jsonpath):
    splits, common = data_world(seed)
    summaries = []
    for ix, method in enumerate(METHODS):
        train_seed = seed * 100 + ix
        model, trainwall, trainloss = train(method, train_seed, splits)
        nbytes, digest, arrays = payload(model, Path(outdir) / f"{split}_{seed}_{method}.npz")
        metrics = evaluate(model, splits, common)
        params = sum(v.numel() for v in model.parameters())
        macs_per_mimo_call = M * ((IN * H) + (H * H) + (H * CLASSES))
        if method == "mirror_givens":
            macs_per_mimo_call += M * 4
        summaries.append({"condition": split, "seed": seed, "method": method, "serialized_bytes": nbytes,
                          "payload_sha256": digest, "parameter_count": params,
                          "train_examples_per_member": UPDATES * BATCH, "total_train_examples": UPDATES * BATCH * M,
                          "optimizer_updates": UPDATES, "train_wall_s": trainwall,
                          "train_final_loss": trainloss, "macs_per_mimo_call_proxy": macs_per_mimo_call,
                          **metrics})
    result = {"condition": split, "seed": seed, "summaries": summaries}
    p = Path(jsonpath)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", choices=["development", "fresh"], required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    run(args.seed, args.split, args.outdir, args.json)
