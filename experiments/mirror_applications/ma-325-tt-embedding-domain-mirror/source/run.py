import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
V = 16
E = 16
R = 4
DOMAINS = 3
ALIGNED = 2
UPDATES = 600
BATCH = 128
LR = 0.01
TEACHER_LOGIT_SCALE = 32.0
RHO = 0.25
METHODS = ["hard_tied", "independent_tt", "shared_direct", "mirror_phase", "shared_private", "independent_full"]


def tt_matrix(g1, g2, g3, g4):
    return torch.einsum("ia,ajb,bkc,cl->ijkl", g1, g2, g3, g4).reshape(V, E)


def init_cores(generator, scale=0.18):
    return [nn.Parameter(torch.randn(shape, generator=generator) * scale) for shape in
            [(4, R), (R, 4, R), (R, 4, R), (R, 4)]]


def teacher_world(seed):
    gen = torch.Generator().manual_seed(seed)
    shared = init_cores(gen, 0.28)
    u = torch.randn(shared[1].shape, generator=gen) * 0.10
    v = torch.randn(shared[1].shape, generator=gen) * 0.10
    angles = [-0.85, 1.15]
    tables = []
    for d in range(ALIGNED):
        cores = [shared[0], shared[1] + RHO * (np.cos(angles[d]) * u + np.sin(angles[d]) * v), shared[2], shared[3]]
        tables.append(tt_matrix(*cores).detach())
    unrelated = init_cores(gen, 0.28)
    tables.append(tt_matrix(*unrelated).detach())
    transitions = []
    for table in tables:
        # Keep the teacher transition distribution learnable from the tied
        # embedding softmax; the extra temperature creates a useful but
        # nontrivial bigram signal without changing the frozen task.
        logits = (table * TEACHER_LOGIT_SCALE) @ (table * TEACHER_LOGIT_SCALE).T / (E ** 0.5)
        transitions.append(torch.softmax(logits, dim=-1))
    return torch.stack(tables), torch.stack(transitions)


def sample_pairs(seed, transitions):
    rng = np.random.default_rng(seed)
    splits = {}
    for split, n, salt in [("train", 32768, 101), ("validation", 4096, 211), ("test", 8192, 307)]:
        xs, ys = [], []
        for d in range(DOMAINS):
            p = transitions[d].cpu().numpy()
            rr = np.random.default_rng(seed + salt + d * 37)
            x = rr.integers(0, V, size=n, dtype=np.int64)
            y = np.asarray([rr.choice(V, p=p[t]) for t in x], dtype=np.int64)
            xs.append(x)
            ys.append(y)
        splits[split] = (torch.tensor(np.stack(xs), dtype=torch.long), torch.tensor(np.stack(ys), dtype=torch.long))
    return splits


class TTBank(nn.Module):
    def __init__(self, method, seed):
        super().__init__()
        self.method = method
        gen = torch.Generator().manual_seed(seed)
        if method == "independent_tt":
            self.tables = nn.ParameterList([p for _ in range(DOMAINS) for p in init_cores(gen)])
        elif method == "independent_full":
            self.full = nn.Parameter(torch.randn(DOMAINS, V, E, generator=gen) * 0.08)
        else:
            self.shared = nn.ParameterList(init_cores(gen))
            if method in ("shared_direct", "mirror_phase", "shared_private"):
                self.u = nn.Parameter(torch.randn(self.shared[1].shape, generator=gen) * 0.05)
                self.v = nn.Parameter(torch.randn(self.shared[1].shape, generator=gen) * 0.05)
            if method == "shared_direct":
                self.coeffs = nn.Parameter(torch.zeros(DOMAINS, 2))
            elif method == "mirror_phase":
                # All logical domains receive a coordinate. Domain 2 is
                # intentionally unrelated in the teacher and tests whether
                # the frozen phase-only family can represent it.
                self.phases = nn.Parameter(torch.zeros(DOMAINS))
            elif method == "shared_private":
                self.coeffs = nn.Parameter(torch.zeros(ALIGNED, 2))
                self.private = nn.ParameterList(init_cores(gen))

    def effective_table(self, domain):
        if self.method == "independent_tt":
            cs = list(self.tables[domain * 4:(domain + 1) * 4])
        elif self.method == "independent_full":
            return self.full[domain]
        elif self.method in ("hard_tied",):
            cs = list(self.shared)
        elif self.method == "shared_private" and domain == 2:
            cs = list(self.private)
        else:
            cs = list(self.shared)
            if self.method == "shared_direct":
                a, b = self.coeffs[domain]
                cs[1] = cs[1] + a * self.u + b * self.v
            elif self.method == "mirror_phase" or (self.method == "shared_private" and domain < ALIGNED):
                phase = self.phases[domain] if self.method == "mirror_phase" else None
                if phase is not None:
                    cs[1] = cs[1] + RHO * (torch.cos(phase) * self.u + torch.sin(phase) * self.v)
                else:
                    a, b = self.coeffs[domain]
                    cs[1] = cs[1] + a * self.u + b * self.v
        return tt_matrix(*cs)

    def forward(self, token_ids):
        outputs = []
        for d in range(DOMAINS):
            table = self.effective_table(d)
            x = token_ids[d]
            outputs.append(table[x] @ table.T)
        return torch.stack(outputs)


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


def inference_payload(model, path):
    arrays = {}
    for k, value in model.state_dict().items():
        a = value.detach().cpu().numpy()
        arrays[k] = a.astype("<f2") if a.dtype.kind == "f" else a
    arrays["domain_ids"] = np.arange(DOMAINS, dtype=np.uint8)
    arrays["meta"] = np.asarray([V, E, R, DOMAINS], dtype=np.uint16)
    n, h = deterministic_pack(arrays, path)
    return n, h, arrays


def load_payload_model(method, arrays, seed):
    model = TTBank(method, seed)
    ref = model.state_dict()
    state = {}
    for k, tensor in ref.items():
        x = torch.from_numpy(np.array(arrays[k], copy=True))
        if tensor.is_floating_point():
            x = x.to(dtype=tensor.dtype)
        state[k] = x
    model.load_state_dict(state)
    return model.eval()


def train(method, seed, splits):
    model = TTBank(method, seed * 100 + METHODS.index(method))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    xtr, ytr = splits["train"]
    rng = np.random.default_rng(seed + 999)
    model.train()
    start = time.perf_counter()
    loss_sum = 0.0
    for _ in range(UPDATES):
        ids = rng.integers(0, len(xtr[0]), size=BATCH)
        logits = model(xtr[:, ids])
        loss = sum(F.cross_entropy(logits[d], ytr[d, ids]) for d in range(DOMAINS)) / DOMAINS
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_sum += float(loss.detach())
    return model.eval(), time.perf_counter() - start, loss_sum / UPDATES


def evaluate(model, split):
    x, y = split
    with torch.no_grad():
        logits = model(x)
        nll = [float(F.cross_entropy(logits[d], y[d])) for d in range(DOMAINS)]
        acc = [float((logits[d].argmax(-1) == y[d]).float().mean()) for d in range(DOMAINS)]
        start = time.perf_counter()
        for _ in range(30):
            model(x)
        wall = (time.perf_counter() - start) / 30
    return {"domain_nll": nll, "domain_accuracy": acc, "macro_nll": float(np.mean(nll)),
            "macro_accuracy": float(np.mean(acc)), "examples_per_s": float(DOMAINS * len(x[0]) / wall),
            "inference_wall_s_per_call": wall}


def run(seed, split, outdir, jsonpath):
    teacher, trans = teacher_world(seed)
    splits = sample_pairs(seed, trans)
    summaries = []
    for method in METHODS:
        train_seed = seed * 100 + METHODS.index(method)
        trained, trainwall, trainloss = train(method, seed, splits)
        path = Path(outdir) / f"{split}_{seed}_{method}.npz"
        n, digest, arrays = inference_payload(trained, path)
        model = load_payload_model(method, arrays, train_seed)
        val = evaluate(model, splits["validation"])
        metrics = evaluate(model, splits["test"])
        params = sum(p.numel() for p in trained.parameters())
        macs = DOMAINS * (E * E + V * E)
        summaries.append({"condition": split, "seed": seed, "method": method, "serialized_bytes": n,
                          "payload_sha256": digest, "parameter_count": params, "optimizer_updates": UPDATES,
                          "train_pairs_per_domain": UPDATES * BATCH, "total_train_pairs": UPDATES * BATCH * DOMAINS,
                          "train_wall_s": trainwall, "train_final_loss": trainloss, "macs_per_batch_proxy": macs,
                          "validation_domain_nll": val["domain_nll"], "domain_nll": metrics["domain_nll"],
                          "domain_accuracy": metrics["domain_accuracy"], "macro_nll": metrics["macro_nll"],
                          "macro_accuracy": metrics["macro_accuracy"], "examples_per_s": metrics["examples_per_s"],
                          "inference_wall_s_per_call": metrics["inference_wall_s_per_call"],
                          "tied_embedding_softmax": True})
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
