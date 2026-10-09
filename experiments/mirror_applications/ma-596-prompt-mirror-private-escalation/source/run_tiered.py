#!/usr/bin/env python3
"""MA-596 sequential shared-prompt / Mirror-code / private-residual screen."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

REV = 'e93a9faa9c77e5d09219f6c868bfc7a1bd65593c'
MODEL_SHA = '3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd'
DATA_SHA = {
    'train.txt': '9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f',
    'test.txt': 'd790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0',
}
N_TASKS, PROMPT_LEN, HIDDEN, RANK, WIN = 8, 8, 512, 4, 64
THRESHOLD = 0.05


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def articles(path: Path, tok):
    found, title, body = [], None, []
    def flush():
        if title:
            ids = np.asarray(tok.encode(title + '\n\n' + '\n'.join(body), add_special_tokens=False), dtype=np.int64)
            if len(ids) >= 2048:
                found.append((title, ids))
    for line in path.read_text(encoding='utf8').splitlines():
        if line.startswith(' = ') and line.endswith(' = ') and not line.startswith(' = ='):
            flush()
            title, body = line.strip(' ='), []
        elif title:
            body.append(line)
    flush()
    return found


def canonical_pca(prompts: np.ndarray, rank: int = RANK):
    """Centered PCA; deterministic sign convention, returned as float32."""
    mean = prompts.mean(axis=0)
    centered = prompts - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:rank].T.copy()
    if basis.shape[1] < rank:
        basis = np.pad(basis, ((0, 0), (0, rank - basis.shape[1])))
    for j in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, j])))
        if basis[pivot, j] < 0:
            basis[:, j] *= -1
    codes = centered @ basis
    recon = mean + codes @ basis.T
    return mean.astype(np.float32), basis.astype(np.float32), codes.astype(np.float32), recon.astype(np.float32)


def serialize_payload(path: Path, mean, basis, codes, private, order, schema=596):
    # Every inference object, task order, index table and schema tag is paid.
    arrays = {
        'mean': np.asarray(mean, np.float16), 'basis': np.asarray(basis, np.float16),
        'codes': np.asarray(codes, np.float16), 'task_order': np.asarray(order, np.int32),
        'shape': np.asarray([len(order), PROMPT_LEN, HIDDEN, RANK], np.int32),
        'schema': np.asarray([schema, 1], np.int32),
    }
    indices = np.flatnonzero(np.any(np.asarray(private) != 0, axis=(1, 2)))
    if len(indices):
        arrays['private_indices'] = indices.astype(np.int32)
        arrays['private_values'] = np.asarray(private[indices], np.float16)
    return np.savez(path, **arrays)


def train_prompt(model, embed, ids: np.ndarray, seed: int, rng: np.random.Generator):
    torch.manual_seed(seed)
    prompt = torch.nn.Parameter(torch.randn(PROMPT_LEN, HIDDEN) * 0.02)
    opt = torch.optim.AdamW([prompt], lr=0.05)
    updates = 0
    for _ in range(8):
        losses = []
        for _ in range(4):
            start = int(rng.integers(0, 1024 - WIN))
            x = torch.as_tensor(ids[start:start + WIN], dtype=torch.long)
            inp = torch.cat([prompt[None], embed(x)[None]], dim=1)
            labels = torch.full((1, PROMPT_LEN + len(x)), -100, dtype=torch.long)
            labels[0, PROMPT_LEN:] = x
            losses.append(model(inputs_embeds=inp, labels=labels, use_cache=False, return_dict=True).loss)
        loss = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        updates += 1
    return prompt.detach().cpu().numpy().astype(np.float32), updates


def eval_prompt(model, embed, prompt: np.ndarray, windows):
    vals = []
    p = torch.as_tensor(prompt, dtype=torch.float32)[None]
    with torch.inference_mode():
        for ids in windows:
            x = torch.as_tensor(ids, dtype=torch.long)
            inp = torch.cat([p, embed(x)[None]], dim=1)
            labels = torch.full((1, PROMPT_LEN + len(x)), -100, dtype=torch.long)
            labels[0, PROMPT_LEN + WIN:] = x[WIN:]
            out = model(inputs_embeds=inp, labels=labels, use_cache=False, return_dict=True)
            vals.append(float(out.loss))
    return float(np.mean(vals))


def task_windows(ids: np.ndarray):
    # Four deterministic validation windows, fully inside the held-out article suffix.
    starts = [1024, 1152, 1408, 1664]
    return [ids[s:s + 128] for s in starts]


def replay_sequence(targets, scores_full, windows, model, embed, outdir: Path):
    n = len(targets)
    trace, cumulative_encode_s = [], 0.0
    final = None
    total_macs_proxy = 0
    for seen in range(1, n + 1):
        t0 = time.perf_counter()
        mean, basis, codes, recon = canonical_pca(np.stack(targets[:seen]))
        # Evaluate the exact FP16 inference representation that will be serialized.
        mean = mean.astype(np.float16).astype(np.float32)
        basis = basis.astype(np.float16).astype(np.float32)
        codes = codes.astype(np.float16).astype(np.float32)
        recon = mean + codes @ basis.T
        cumulative_encode_s += time.perf_counter() - t0
        # Count one dense projection and one decode per seen task; SVD fit cost is separately approximated.
        total_macs_proxy += 2 * seen * PROMPT_LEN * HIDDEN * RANK + 2 * HIDDEN * PROMPT_LEN * RANK * RANK
        # Evaluate current reconstructions on all old and new tasks before private fallback.
        mirror_nll = [eval_prompt(model, embed, recon[i], windows[i]) for i in range(seen)]
        escalated = [mirror_nll[i] - scores_full[i] > THRESHOLD for i in range(seen)]
        decoded = recon.copy()
        private = np.zeros_like(decoded)
        for i, use_private in enumerate(escalated):
            if use_private:
                private[i] = (np.asarray(targets[i], np.float32) - recon[i]).astype(np.float16).astype(np.float32)
                decoded[i] += private[i]
        tiered_nll = [eval_prompt(model, embed, decoded[i], windows[i]) for i in range(seen)]
        drift = [eval_prompt(model, embed, decoded[i], windows[i]) - scores_full[i] for i in range(seen)]
        trace.append({
            'tasks_seen': seen,
            'private_count': int(sum(escalated)),
            'private_task_indices': [i for i, x in enumerate(escalated) if x],
            'mean_rank4_delta_nll': float(np.mean(np.asarray(mirror_nll) - np.asarray(scores_full[:seen]))),
            'max_tiered_delta_nll': float(max(drift)),
            'old_task_max_drift': float(max(drift[:-1], default=0.0)),
            'cumulative_reencode_seconds': cumulative_encode_s,
        })
        final = (mean, basis, codes, private, decoded, mirror_nll, tiered_nll, escalated)
    assert final is not None
    mean, basis, codes, private, decoded, mirror_nll, tiered_nll, escalated = final
    pth = outdir / 'tiered_payload.npz'
    serialize_payload(pth, mean, basis, codes, private, list(range(n)))
    payload_bytes = pth.stat().st_size
    # Native control uses the identical PCA coordinates. Save and hash for exact alias audit.
    native_path = outdir / 'native_rank4_payload.npz'
    serialize_payload(native_path, mean, basis, codes, np.zeros_like(private), list(range(n)), schema=596)
    full_path = outdir / 'independent_payload.npz'
    np.savez(full_path, prompts=np.asarray(targets, np.float16), task_order=np.arange(n, dtype=np.int32),
             shape=np.asarray([n, PROMPT_LEN, HIDDEN], np.int32), schema=np.asarray([598, 1], np.int32))
    shared_path = outdir / 'shared_only_payload.npz'
    np.savez(shared_path, mean=np.asarray(mean, np.float16), task_order=np.arange(n, dtype=np.int32),
             shape=np.asarray([n, PROMPT_LEN, HIDDEN], np.int32), schema=np.asarray([599, 1], np.int32))
    t0 = time.perf_counter()
    mean_inf = mean.astype(np.float16).astype(np.float32)
    basis_inf = basis.astype(np.float16).astype(np.float32)
    codes_inf = codes.astype(np.float16).astype(np.float32)
    private_inf = private.astype(np.float16).astype(np.float32)
    mirror_prompts = mean_inf + codes_inf @ basis_inf.T
    mirror_nll_final = [eval_prompt(model, embed, mirror_prompts[i], windows[i]) for i in range(n)]
    mirror_inference_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    tiered_nll_final = [eval_prompt(model, embed, decoded[i], windows[i]) for i in range(n)]
    tiered_inference_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    shared_nll_final = [eval_prompt(model, embed, mean_inf, w) for w in windows]
    shared_inference_s = time.perf_counter() - t0
    return {
        'trace': trace,
        'final': {
            'full_nll': list(map(float, scores_full)),
            'shared_only_nll': shared_nll_final,
            'mirror_rank4_nll': mirror_nll_final,
            'tiered_nll': tiered_nll_final,
            'delta_vs_full': (np.asarray(tiered_nll_final) - np.asarray(scores_full)).tolist(),
            'mirror_delta_vs_full': (np.asarray(mirror_nll_final) - np.asarray(scores_full)).tolist(),
            'private_count': int(sum(escalated)),
            'private_fraction': float(np.mean(escalated)),
            'max_old_task_drift': float(max((r['old_task_max_drift'] for r in trace), default=0.0)),
            'mirror_bytes': payload_bytes,
            'native_rank4_bytes': native_path.stat().st_size,
            'independent_bytes': full_path.stat().st_size,
            'shared_only_bytes': shared_path.stat().st_size,
            'mirror_sha256': sha(pth),
            'native_rank4_sha256': sha(native_path),
            'independent_sha256': sha(full_path),
            'mirror_inference_payload_bytes_per_task': payload_bytes / n,
            'shared_only_inference_seconds': shared_inference_s,
            'mirror_rank4_inference_seconds': mirror_inference_s,
            'tiered_inference_seconds': tiered_inference_s,
            'reencode_macs_proxy': int(total_macs_proxy),
        },
        'reencode_seconds': cumulative_encode_s,
    }


def run(seed: int, split: str, model_dir: Path, data_dir: Path, outdir: Path):
    t_wall = time.perf_counter()
    torch.set_num_threads(4)
    model_dir, data_dir = Path(model_dir), Path(data_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    if sha(model_dir / 'model.safetensors') != MODEL_SHA:
        raise ValueError('Pythia weight SHA-256 mismatch')
    data_name = 'train.txt' if split == 'dev' else 'test.txt'
    if sha(data_dir / data_name) != DATA_SHA[data_name]:
        raise ValueError(f'{data_name} SHA-256 mismatch')
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    t_load = time.perf_counter()
    model = GPTNeoXForCausalLM.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.float32).eval()
    model_load_seconds = time.perf_counter() - t_load
    for p in model.parameters():
        p.requires_grad_(False)
    pool = articles(data_dir / data_name, tok)
    if len(pool) < N_TASKS:
        raise ValueError(f'only {len(pool)} eligible articles')
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(pool), size=N_TASKS, replace=False)
    tasks = [pool[int(i)] for i in chosen]
    embed = model.get_input_embeddings()
    targets, windows, titles, train_seconds = [], [], [], []
    updates = 0
    for ti, (title, ids) in enumerate(tasks):
        t0 = time.perf_counter()
        prompt, count = train_prompt(model, embed, ids, seed * 100 + ti, rng)
        train_seconds.append(time.perf_counter() - t0)
        updates += count
        targets.append(prompt)
        windows.append(task_windows(ids))
        titles.append(title)
    t_eval = time.perf_counter()
    t_full_eval = time.perf_counter()
    targets = [p.astype(np.float16).astype(np.float32) for p in targets]
    full_nll = [eval_prompt(model, embed, p, w) for p, w in zip(targets, windows)]
    full_eval_seconds = time.perf_counter() - t_full_eval
    full_eval_s = time.perf_counter() - t_eval
    seq = replay_sequence(targets, full_nll, windows, model, embed, outdir)
    rep = {
        'experiment_id': 'MA-596', 'seed': seed, 'split': split, 'model_revision': REV,
        'model_sha256': MODEL_SHA, 'dataset_file': data_name, 'dataset_sha256': DATA_SHA[data_name],
        'task_titles': titles, 'task_count': N_TASKS, 'prompt_length': PROMPT_LEN,
        'hidden_size': HIDDEN, 'rank': RANK, 'private_threshold_nat_per_token': THRESHOLD,
        'optimizer_updates': updates, 'mean_task_training_seconds': float(np.mean(train_seconds)),
        'total_task_training_seconds': float(sum(train_seconds)), 'full_prompt_eval_seconds': full_eval_s,
        'model_load_seconds': model_load_seconds,
        'sequence': seq['final'], 'retention_trace': seq['trace'],
        'cumulative_reencode_seconds': seq['reencode_seconds'],
        'compute': {'device': 'cpu', 'torch_threads': torch.get_num_threads(), 'virtual_tokens_per_example': PROMPT_LEN,
                    'added_prompt_embedding_ops_per_example': PROMPT_LEN * HIDDEN,
                    'full_prompt_inference_seconds': full_eval_seconds,
                    'total_wall_seconds': time.perf_counter() - t_wall},
    }
    rep['common_backbone_bytes'] = int((model_dir / 'model.safetensors').stat().st_size)
    rep['full_deployment_bytes'] = rep['common_backbone_bytes'] + seq['final']['mirror_bytes']
    rep['independent_full_deployment_bytes'] = rep['common_backbone_bytes'] + seq['final']['independent_bytes']
    (outdir / 'metrics.json').write_text(json.dumps(rep, indent=2, sort_keys=True) + '\n')
    print(json.dumps(rep, indent=2, sort_keys=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--split', choices=['dev', 'fresh'], required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    run(a.seed, a.split, a.model_dir, a.data_dir, a.out)

if __name__ == '__main__':
    main()
