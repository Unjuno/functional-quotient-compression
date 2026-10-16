#!/usr/bin/env python3
"""Rebuild the three frozen T277 controls from a trusted user checkpoint on CPU.
No tuning, downloads, checkpoint changes, or baseline-hash replacement.
MPS/CUDA fit is deliberately NOT offered: historical moments use float64 on CPU.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, platform, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
LANE = ROOT / 'experiments/t282'
sys.path.insert(0, str(LANE / 'code'))
from engine import BPETokenizer, forward, load_checkpoint, tensor_hash
from binary_codec import encode_uniform, fp16, read_model, write_model
from affine_refit import encode_refit
from artifact_gate import checked_decode


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path: Path, value: dict) -> None:
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def collect_cpu(state: dict, config: dict, probes: list) -> dict:
    """Same accumulation order and arithmetic as historical run_experiment.collect."""
    mapping = {v.data_ptr(): k for k, v in state.items() if v.ndim == 2}
    sums, counts = {}, {}
    original_linear = F.linear
    def observed(x, weight, bias=None):
        name = mapping.get(weight.data_ptr())
        if name and ('.mlp.' in name or '.attn.' in name):
            flat = x.detach().reshape(-1, x.shape[-1])
            ss = (flat.double() * flat.double()).sum(0).cpu()
            if name not in sums:
                sums[name], counts[name] = ss, len(flat)
            else:
                sums[name] += ss
                counts[name] += len(flat)
        return original_linear(x, weight, bias)
    F.linear = observed
    try:
        with torch.inference_mode():
            for probe in probes:
                forward(state, config, torch.tensor([probe['input_ids']], device='cpu'))
    finally:
        F.linear = original_linear
    result = {}
    for name, total in sums.items():
        a = (total / counts[name]).numpy()
        a = np.maximum(a, max(float(a.mean()) * 1e-8, 1e-12))
        a /= a.mean()
        result[name] = a.astype(np.float32)
    if len(result) != 6 * config['num_layers']:
        raise AssertionError('Incomplete activation-moment collection')
    return result


def main() -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("The T282 local lane requires Python 3.11 or newer.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', action='append', help='Repeat to select frozen candidates; default: all three.')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a NEW output directory')
    lock = json.loads((LANE / 'locks/frozen_candidates.json').read_text())
    records = lock['candidates']
    if args.candidate:
        if len(set(args.candidate)) != len(args.candidate) or set(args.candidate) - {r['candidate'] for r in records}:
            raise ValueError('Unknown or duplicate frozen candidate')
        records = [r for r in records if r['candidate'] in args.candidate]
    checkpoint = args.models_root / lock['checkpoint']
    if sha(checkpoint) != lock['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch: do not silently replace the source model')
    data = LANE / 'data/calibration_probes.json'
    if sha(data) != lock['calibration_sha256']:
        raise ValueError('Frozen calibration subset changed')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(lock['fit_seed'])
    cfg, state, raw = load_checkpoint(checkpoint.parent, 'cpu')
    probes = json.loads(data.read_text())
    tokenizer = BPETokenizer(checkpoint.parent)
    if not all(tokenizer.encode(p['text']) == p['input_ids'] for p in probes):
        raise AssertionError('Tokenizer/calibration token-ID mismatch')
    args.output.mkdir(parents=True)
    save(args.output / 'protocol.json', {'kind': 'H282_PACKAGING_REBUILD_NOT_NEW_SCIENTIFIC_EXPERIMENT',
         'candidates': records, 'checkpoint_sha256': lock['checkpoint_sha256'], 'calibration_sha256': sha(data),
         'environment': {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
                         'numpy': np.__version__, 'backend': 'cpu', 'threads': 2, 'batch': 1, 'dtype': 'float32'},
         'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__).resolve(), *sorted((LANE / 'code').glob('*.py'))]},
         'argv': sys.argv,
         'cross_platform_bitwise_equality': 'test outcome, not an assumption',
         'input_policy': 'Trusted supplied checkpoint; torch.load(weights_only=True); no network'})
    moments = collect_cpu(state, cfg, probes) if any(r['method'] == 'act' for r in records) else {}
    rows = []
    for expected in records:
        name = expected['candidate']
        print('REBUILD', name, flush=True)
        sections = []
        for key, tensor in state.items():
            a = tensor.numpy()
            if a.ndim == 1:
                desc, blob = {'kind': 'fp16', 'shape': list(a.shape)}, fp16(a).tobytes()
            elif expected['method'] == 'rtn':
                desc, blob = encode_uniform(a, expected['bits'], expected['group'])
            else:
                desc, blob, _ = encode_refit(a, expected['bits'], moments.get(key), group=expected['group'])
            sections.append(({**desc, 'name': key}, blob))
        path = args.output / (name + '.fqc')
        info = write_model(path, cfg, sections, {'experiment': 'T277', 'method': expected['method'],
                           'bits': expected['bits'], 'group': expected['group'],
                           'checkpoint_sha256': lock['checkpoint_sha256']})
        del sections
        c1, s1, _ = checked_decode(path)
        c2, s2, _ = read_model(path)
        parity = c1 == c2 and set(s1) == set(s2) and all(torch.equal(s1[k], s2[k]) for k in s1)
        row = {'candidate': name, **info, 'decoded_tensor_sha256': tensor_hash(s1),
               'independent_decoder_equal': parity, 'tensors': len(s1)}
        row['historical_bytes_exact'] = info['sha256'] == expected['sha256'] and info['bytes'] == expected['bytes']
        row['historical_tensors_exact'] = row['decoded_tensor_sha256'] == expected['decoded_tensor_sha256']
        row['PASS'] = row['historical_bytes_exact'] and row['historical_tensors_exact'] and parity
        save(args.output / (name + '.verification.json'), row)
        rows.append(row)
        del s1, s2
        gc.collect()
    passed = all(r['PASS'] for r in rows)
    save(args.output / 'REBUILD_RESULT.json', {'status': 'PASS' if passed else 'FAIL', 'rows': rows,
         'scope': 'artifact generation and two decoders, NOT new quality evaluation or official runtime parity'})
    if not passed:
        raise SystemExit('Rebuild mismatch recorded. Keep evidence and investigate; never rewrite reference hashes.')
    print('PASS: historical artifact bytes and tensors reproduced', flush=True)


if __name__ == '__main__':
    main()
