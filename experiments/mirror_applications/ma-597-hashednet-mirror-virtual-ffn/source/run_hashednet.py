#!/usr/bin/env python3
"""MA-597 hashed first-layer FFN, Mirror Givens View, and native controls."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F

INPUT, HIDDEN, OUTPUT = 64, 128, 10
UPDATES, BATCH, LR = 800, 128, 0.01
METHODS = ('dense', 'hash_2048', 'hash_4096', 'mirror_givens', 'diagonal_gate', 'rank1_residual')


def digits_data():
    ds = load_digits()
    x = np.ascontiguousarray(ds.data.astype(np.float32) / 16.0)
    y = np.ascontiguousarray(ds.target.astype(np.int64))
    raw = np.ascontiguousarray(ds.data.astype(np.float32))
    digest = hashlib.sha256(raw.tobytes() + y.tobytes()).hexdigest()
    return x, y, digest


def hash_map(bucket_count: int, seed: int, shape=(INPUT, HIDDEN)):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, bucket_count, size=shape, dtype=np.int64)
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=shape)
    return indices, signs


def collision_metrics(indices, bucket_count):
    counts = np.bincount(indices.reshape(-1), minlength=bucket_count)
    used = int(np.count_nonzero(counts))
    n = int(indices.size)
    pairs = int(np.sum(counts * (counts - 1) // 2))
    return {
        'unique_bucket_count': used,
        'unique_bucket_fraction': used / n,
        'collision_rate': 1.0 - used / n,
        'pairwise_collision_count': pairs,
        'maximum_bucket_load': int(counts.max(initial=0)),
    }


def givens_rotate(x: torch.Tensor, angles: torch.Tensor):
    """Apply 32 disjoint adjacent-pair rotations on the last 64-dimension axis."""
    even, odd = x[..., 0::2], x[..., 1::2]
    c, s = torch.cos(angles), torch.sin(angles)
    a = c * even - s * odd
    b = s * even + c * odd
    out = torch.empty_like(x)
    out[..., 0::2], out[..., 1::2] = a, b
    return out


def new_params(method, bucket_count, seed):
    torch.manual_seed(seed)
    params = {}
    if method == 'dense':
        params['w1'] = torch.nn.Parameter(torch.empty(INPUT, HIDDEN))
        torch.nn.init.kaiming_uniform_(params['w1'].T, a=0.0, nonlinearity='relu')
    else:
        std = (2.0 / INPUT) ** 0.5
        params['theta'] = torch.nn.Parameter(torch.randn(bucket_count) * std)
    params['b1'] = torch.nn.Parameter(torch.zeros(HIDDEN))
    params['w2'] = torch.nn.Parameter(torch.empty(HIDDEN, OUTPUT))
    params['b2'] = torch.nn.Parameter(torch.zeros(OUTPUT))
    torch.nn.init.xavier_uniform_(params['w2'].T)
    if method == 'mirror_givens':
        params['angles'] = torch.nn.Parameter(torch.zeros(INPUT // 2))
    elif method == 'diagonal_gate':
        params['scale'] = torch.nn.Parameter(torch.ones(INPUT))
    elif method == 'rank1_residual':
        params['u'] = torch.nn.Parameter(torch.randn(INPUT, 1) * 0.01)
        params['v'] = torch.nn.Parameter(torch.randn(1, HIDDEN) * 0.01)
    return params


def current_w1(params, method, indices, signs):
    if method == 'dense':
        return params['w1']
    w = params['theta'][torch.as_tensor(indices)] * torch.as_tensor(signs)
    w = w.reshape(INPUT, HIDDEN)
    if method == 'rank1_residual':
        w = w + params['u'] @ params['v']
    return w


def logits(params, method, x, indices=None, signs=None):
    w1 = current_w1(params, method, indices, signs)
    if method == 'mirror_givens':
        x = givens_rotate(x, params['angles'])
    elif method == 'diagonal_gate':
        x = x * params['scale']
    h = F.relu(x @ w1 + params['b1'])
    return h @ params['w2'] + params['b2']


def train_method(method, bucket_count, x_train, y_train, run_seed, method_index, hash_seed):
    params = new_params(method, bucket_count, run_seed * 100 + method_index)
    opt = torch.optim.AdamW(list(params.values()), lr=LR, weight_decay=0.0)
    if method == 'dense':
        indices = signs = None
    else:
        indices, signs = hash_map(bucket_count, hash_seed)
    rng = np.random.default_rng(run_seed * 1000 + method_index)
    xt = torch.as_tensor(x_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.long)
    start = time.perf_counter()
    loss_trace = []
    for step in range(UPDATES):
        batch_ids = rng.integers(0, len(xt), size=BATCH)
        xb, yb = xt[batch_ids], yt[batch_ids]
        loss = F.cross_entropy(logits(params, method, xb, indices, signs), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in (0, UPDATES - 1):
            loss_trace.append(float(loss.detach()))
    elapsed = time.perf_counter() - start
    return params, indices, signs, elapsed, loss_trace


def state_arrays(params, method, bucket_count, run_seed, hash_seed):
    arr = {k: v.detach().cpu().numpy().astype(np.float16) for k, v in params.items()}
    method_bytes = np.frombuffer(method.encode('ascii'), dtype=np.uint8)
    arr.update({
        'method_ascii': method_bytes,
        'bucket_count': np.asarray([bucket_count if method != 'dense' else 0], dtype=np.int32),
        'hash_seed': np.asarray([hash_seed if method != 'dense' else 0], dtype=np.uint32),
        'run_seed': np.asarray([run_seed], dtype=np.uint32),
        'shape': np.asarray([INPUT, HIDDEN, OUTPUT], dtype=np.int32),
        'schema': np.asarray([597, 1], dtype=np.int32),
    })
    return arr


def load_state(path):
    with np.load(path, allow_pickle=False) as z:
        arr = {k: z[k].copy() for k in z.files}
    method = bytes(arr.pop('method_ascii').tolist()).decode('ascii')
    bucket_count = int(arr.pop('bucket_count')[0])
    hash_seed = int(arr.pop('hash_seed')[0])
    arr.pop('run_seed'); arr.pop('shape'); arr.pop('schema')
    params = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in arr.items()}
    if method == 'dense':
        indices = signs = None
    else:
        indices, signs = hash_map(bucket_count, hash_seed)
    return method, params, indices, signs


def evaluate_payload(path, x_test, y_test):
    method, params, indices, signs = load_state(path)
    x = torch.as_tensor(x_test, dtype=torch.float32)
    y = torch.as_tensor(y_test, dtype=torch.long)
    torch.set_num_threads(1)
    start = time.perf_counter()
    losses, correct, count = [], 0, 0
    with torch.inference_mode():
        for i in range(0, len(x), BATCH):
            xb, yb = x[i:i+BATCH], y[i:i+BATCH]
            out = logits(params, method, xb, indices, signs)
            losses.append(float(F.cross_entropy(out, yb, reduction='sum')))
            correct += int((out.argmax(dim=1) == yb).sum())
            count += len(yb)
    infer_s = time.perf_counter() - start
    return {'accuracy': correct / count, 'mean_cross_entropy': sum(losses) / count,
            'inference_seconds': infer_s, 'examples_per_second': count / infer_s,
            'count': count}


def effective_w1(payload):
    method, params, indices, signs = load_state(payload)
    w = current_w1(params, method, indices, signs)
    if method == 'mirror_givens':
        # W_eff = G^T W so that x^T W_eff = (Gx)^T W.
        w = givens_rotate(w.T, -params['angles']).T
    elif method == 'diagonal_gate':
        w = params['scale'][:, None] * w
    return w.detach().cpu().numpy()


def save_payload(path, params, method, bucket_count, run_seed, hash_seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **state_arrays(params, method, bucket_count, run_seed, hash_seed))
    return path.stat().st_size


def run(seed: int, split: str, outdir: Path):
    torch.set_num_threads(1)
    x, y, dataset_hash = digits_data()
    if dataset_hash != 'faf2d98f61250c1cdc0b114323133d3c58da04f5c5d28bb78e4bde3c936a75b1':
        raise ValueError('load_digits tensor SHA-256 mismatch')
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    start_all = time.perf_counter()
    hash_seed = (seed * 7919 + 120) & 0xFFFFFFFF
    # Dense teacher is trained first so collision projection error uses an independently fit reference.
    results = []
    teacher_w1 = None
    for mi, method in enumerate(METHODS):
        bucket_count = 4096 if method == 'hash_4096' else 2048
        params, idx, signs, train_s, loss_trace = train_method(method, bucket_count, xtr, ytr, seed, mi, hash_seed)
        payload = outdir / f'{method}.npz'
        byte_count = save_payload(payload, params, method, bucket_count, seed, hash_seed)
        ev = evaluate_payload(payload, xte, yte)
        if method == 'dense':
            teacher_w1 = effective_w1(payload)
        w_eff = effective_w1(payload)
        collision = {}
        projection_rmse = None
        if method != 'dense':
            idx, sign = hash_map(bucket_count, hash_seed)
            collision = collision_metrics(idx, bucket_count)
            if teacher_w1 is not None:
                projection_rmse = float(np.sqrt(np.mean((w_eff - teacher_w1) ** 2)))
        logical_macs = INPUT * HIDDEN + HIDDEN * OUTPUT
        extra_ops = 0
        expansion_lookups = 0
        if method != 'dense':
            expansion_lookups = INPUT * HIDDEN
        if method == 'mirror_givens': extra_ops = (INPUT // 2) * 6
        elif method == 'diagonal_gate': extra_ops = INPUT
        elif method == 'rank1_residual': extra_ops = INPUT + HIDDEN
        metrics = {
            'method': method, 'bucket_count': bucket_count if method != 'dense' else None,
            'serialized_bytes': byte_count, 'payload_sha256': hashlib.sha256(payload.read_bytes()).hexdigest(),
            'train_examples': int(len(xtr)), 'test_examples': int(len(xte)),
            'optimizer_updates': UPDATES, 'examples_seen': UPDATES * BATCH,
            'initial_and_final_train_loss': loss_trace, 'training_seconds': train_s,
            'test_accuracy': ev['accuracy'], 'test_cross_entropy': ev['mean_cross_entropy'],
            'inference_seconds': ev['inference_seconds'], 'inference_examples_per_second': ev['examples_per_second'],
            'base_macs_per_example': logical_macs, 'extra_view_ops_per_example': extra_ops,
            'hash_expansion_lookups_per_model_load': expansion_lookups,
            'dense_teacher_projection_rmse': projection_rmse,
            'collision': collision,
        }
        results.append(metrics)
    # Teacher-trained layer is the reference for a bucket-tie collision diagnostic.
    # Compute the optimal signed bucket projection of its dense weight matrix.
    dense_w = np.asarray(teacher_w1)
    for row in results:
        if row['method'] == 'dense':
            continue
        b = int(row['bucket_count'])
        ix, sg = hash_map(b, hash_seed)
        sums = np.bincount(ix.reshape(-1), weights=(sg * dense_w).reshape(-1), minlength=b)
        counts = np.bincount(ix.reshape(-1), minlength=b)
        theta = sums / np.maximum(counts, 1)
        projected = (theta[ix] * sg)
        row['dense_teacher_best_hash_projection_rmse'] = float(np.sqrt(np.mean((projected - dense_w) ** 2)))
        row['collision']['bucket_pair_count'] = row['collision'].pop('pairwise_collision_count')
    report = {
        'experiment_id':'MA-597', 'seed':seed, 'split':split, 'dataset':'sklearn.load_digits',
        'sklearn_version':sklearn.__version__, 'dataset_tensor_sha256':dataset_hash,
        'split_random_state':seed, 'stratified':True, 'architecture':[INPUT,HIDDEN,OUTPUT],
        'run_hash_seed':hash_seed, 'methods':results,
        'common_compute':{'device':'cpu','torch_threads':1,'updates_per_method':UPDATES,
                          'input_examples_seen_per_method':UPDATES*BATCH,'total_wall_seconds':time.perf_counter()-start_all},
    }
    (outdir/'metrics.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2,sort_keys=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--split',choices=['dev','fresh'],required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.seed,a.split,a.out)
if __name__=='__main__':main()
