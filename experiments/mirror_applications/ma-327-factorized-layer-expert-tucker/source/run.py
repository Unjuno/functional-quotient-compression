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
LAYERS = EXPERTS = 4
IN = OUT = 16
RANK = 2
BASIS = RANK
UPDATES = 500
BATCH = 64
LR = 0.025
METHODS = ["hard_tied", "flat_pair", "ordinary_product", "mirror_product", "independent_full"]


def world(seed):
    g = torch.Generator().manual_seed(seed)
    basis = torch.randn(BASIS, IN, OUT, generator=g) * 0.35
    layer_code = torch.randn(LAYERS, RANK, generator=g)
    expert_code = torch.randn(EXPERTS, RANK, generator=g)
    coeff = torch.einsum("lr,er->ler", layer_code, expert_code)
    targets = torch.einsum("lek,kio->leio", coeff, basis)
    # Normalize target maps to stabilize fitting while preserving a nontrivial
    # held-out combination task.
    targets = targets / targets.std() * 0.35
    return basis, coeff, targets


def split_pairs(seed, targets):
    pairs = {}
    for name, n, salt in [("train", 512, 101), ("validation", 256, 211), ("test", 1024, 307)]:
        xs, ys = [], []
        for l in range(LAYERS):
            for e in range(EXPERTS):
                r = np.random.default_rng(seed + salt + l * 41 + e * 79)
                x = torch.tensor(r.standard_normal((n, IN)).astype(np.float32))
                y = x @ targets[l, e]
                xs.append(x)
                ys.append(y)
        pairs[name] = (torch.stack(xs).reshape(LAYERS, EXPERTS, n, IN),
                       torch.stack(ys).reshape(LAYERS, EXPERTS, n, OUT))
    return pairs


class Bank(torch.nn.Module):
    def __init__(self, method, seed, basis_init):
        super().__init__()
        self.method = method
        g = torch.Generator().manual_seed(seed)
        self.basis = torch.nn.Parameter(basis_init.clone())
        if method == "flat_pair":
            self.coeffs = torch.nn.Parameter(torch.zeros(LAYERS, EXPERTS, BASIS))
        elif method in ("ordinary_product", "mirror_product"):
            self.layer = torch.nn.Parameter(torch.randn(LAYERS, RANK, generator=g) * 0.1)
            self.expert = torch.nn.Parameter(torch.randn(EXPERTS, RANK, generator=g) * 0.1)
        elif method == "independent_full":
            self.full = torch.nn.Parameter(torch.randn(LAYERS, EXPERTS, IN, OUT, generator=g) * 0.1)

    def matrix(self, l, e):
        if self.method == "independent_full":
            return self.full[l, e]
        if self.method == "hard_tied":
            c = torch.ones(BASIS)
        elif self.method == "flat_pair":
            c = self.coeffs[l, e]
        else:
            c = self.layer[l] * self.expert[e]
        return torch.einsum("r,rio->io", c, self.basis)

    def forward_pair(self, l, e, x):
        return x @ self.matrix(l, e)


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
    b = path.read_bytes()
    return len(b), hashlib.sha256(b).hexdigest()


def serialize(model, path):
    arr = {k: v.detach().cpu().numpy().astype("<f2") for k, v in model.state_dict().items()}
    arr["meta"] = np.asarray([LAYERS, EXPERTS, IN, OUT, BASIS, RANK], dtype=np.uint16)
    return deterministic_pack(arr, path), arr


def load_model(method, arrays, seed, basis_init):
    model = Bank(method, seed, basis_init)
    state = {k: torch.from_numpy(np.array(arrays[k], copy=True)).to(dtype=v.dtype)
             for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model.eval()


def metric(model, split):
    x, y = split
    errs, denom = [], []
    with torch.no_grad():
        for l in range(LAYERS):
            for e in range(EXPERTS):
                pred = model.forward_pair(l, e, x[l, e])
                errs.append(((pred-y[l, e])**2).mean().item())
                denom.append((y[l, e]**2).mean().item())
    return {"pair_mse": errs, "mean_mse": float(np.mean(errs)), "mean_nmse": float(np.mean(np.array(errs)/np.array(denom)))}


def fit(method, seed, basis_init, coeff_target, pairs, heldout):
    model = Bank(method, seed, basis_init)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    xtr, ytr = pairs["train"]
    start = time.perf_counter()
    for _ in range(UPDATES):
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for l in range(LAYERS):
            for e in range(EXPERTS):
                # Reserve one diagonal pair from training coefficient fitting;
                # evaluate it as a held-out compositional combination.
                if heldout and l == e:
                    continue
                idx = np.random.default_rng(seed + _ * 101 + l*7 + e).integers(0, xtr.shape[2], size=BATCH)
                pred = model.forward_pair(l, e, xtr[l, e, idx])
                loss = loss + ((pred-ytr[l, e, idx])**2).mean()
        loss = loss / (LAYERS * EXPERTS - (LAYERS if heldout else 0))
        loss.backward()
        opt.step()
    return model.eval(), time.perf_counter()-start


def run(seed, split, outdir, jsonpath):
    basis, coeff, targets = world(seed)
    pairs = split_pairs(seed, targets)
    rows = []
    for method in METHODS:
        heldout = method in ("ordinary_product", "mirror_product")
        method_seed = seed if method in ("ordinary_product", "mirror_product") else seed + METHODS.index(method)*13
        model, wall = fit(method, method_seed, basis, coeff, pairs, heldout)
        p = Path(outdir) / f"{split}_{seed}_{method}.zip"
        (n, digest), arrays = serialize(model, p)
        reloaded = load_model(method, arrays, method_seed, basis)
        tr = metric(reloaded, pairs["train"])
        va = metric(reloaded, pairs["validation"])
        te = metric(reloaded, pairs["test"])
        params = sum(v.numel() for v in model.parameters())
        # One input projection and one coefficient-basis contraction per pair.
        mac_proxy = LAYERS * EXPERTS * (IN*OUT + BASIS*IN*OUT)
        rows.append({"condition": split, "seed": seed, "method": method, "serialized_bytes": n,
                     "payload_sha256": digest, "parameter_count": params, "updates": UPDATES,
                     "examples_seen": UPDATES*BATCH*(LAYERS*EXPERTS - (LAYERS if heldout else 0)),
                     "train_wall_s": wall, "active_macs_proxy": mac_proxy,
                     "train_nmse": tr["mean_nmse"], "validation_nmse": va["mean_nmse"],
                     "test_nmse": te["mean_nmse"], "test_pair_nmse": te["pair_mse"],
                     "heldout_test_nmse": float(np.mean([te["pair_mse"][i*EXPERTS+i] for i in range(LAYERS)])) if heldout else None})
    out = {"condition": split, "seed": seed, "summaries": rows}
    Path(jsonpath).parent.mkdir(parents=True, exist_ok=True)
    Path(jsonpath).write_text(json.dumps(out, indent=2)+"\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", choices=["development", "fresh"], required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    run(a.seed, a.split, a.outdir, a.json)
