#!/usr/bin/env python3
"""Verify MA-596 frozen provenance, replay hashes, metrics, and split boundary."""
import hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    protocol = ROOT / 'PROTOCOL.json'
    freeze = json.loads((ROOT / 'FREEZE.json').read_text())
    assert sha(protocol) == freeze['protocol_sha256']
    assert sha(ROOT / 'source/run_tiered.py') == freeze['source_sha256']
    assert freeze['frozen_before_development'] is True
    assert freeze['fresh_seeds_sealed'] == [59611, 59612, 59613]
    replayed = {}
    for seed in (59601, 59602):
        dev = ROOT / f'runs/dev_{seed}'
        replay = ROOT / f'runs/replay_{seed}'
        dm = json.loads((dev / 'metrics.json').read_text())
        rm = json.loads((replay / 'metrics.json').read_text())
        assert dm['task_titles'] == rm['task_titles']
        for name in ('independent_payload.npz', 'shared_only_payload.npz', 'tiered_payload.npz', 'native_rank4_payload.npz'):
            assert sha(dev / name) == sha(replay / name)
            assert (dev / name).stat().st_size == dm['sequence'][{
                'independent_payload.npz':'independent_bytes',
                'shared_only_payload.npz':'shared_only_bytes',
                'tiered_payload.npz':'mirror_bytes',
                'native_rank4_payload.npz':'native_rank4_bytes',
            }[name]]
        with np.load(dev / 'tiered_payload.npz', allow_pickle=False) as mirror, np.load(dev / 'native_rank4_payload.npz', allow_pickle=False) as native:
            for key in ('mean', 'basis', 'codes', 'task_order', 'shape', 'schema'):
                assert np.array_equal(mirror[key], native[key]), key
            assert mirror['mean'].dtype == np.float16 and mirror['basis'].dtype == np.float16 and mirror['codes'].dtype == np.float16
        seq = dm['sequence']
        tiered_delta = np.asarray(seq['tiered_nll']) - np.asarray(seq['full_nll'])
        mirror_delta = np.asarray(seq['mirror_rank4_nll']) - np.asarray(seq['full_nll'])
        ratio = seq['mirror_bytes'] / seq['independent_bytes']
        assert float(tiered_delta.max()) <= 0.05
        assert float(mirror_delta.max()) > 0.05
        assert ratio > 0.75
        assert seq['private_count'] in (1, 2)
        # Deterministic replay is exact for core outputs and payloads; wall-clock fields are excluded.
        assert dm['sequence']['delta_vs_full'] == rm['sequence']['delta_vs_full']
        assert dm['sequence']['private_count'] == rm['sequence']['private_count']
        assert [{k:v for k,v in r.items() if 'seconds' not in k} for r in dm['retention_trace']] == [{k:v for k,v in r.items() if 'seconds' not in k} for r in rm['retention_trace']]
        replayed[str(seed)] = {
            'payload_sha256': {name:sha(dev/name) for name in ('independent_payload.npz','shared_only_payload.npz','tiered_payload.npz','native_rank4_payload.npz')},
            'payload_replay_exact': True,
            'mean_tiered_max_delta_nll': float(tiered_delta.max()),
            'mirror_max_delta_nll': float(mirror_delta.max()),
            'tiered_byte_ratio': float(ratio),
            'private_count': int(seq['private_count']),
        }
    assert not (ROOT / 'runs/fresh').exists()
    result = {
        'experiment_id':'MA-596', 'status':'FAIL', 'protocol_sha256':freeze['protocol_sha256'],
        'source_sha256':freeze['source_sha256'], 'development_seeds':[59601,59602],
        'fresh_seeds_accessed':[], 'fresh_data_artifacts_present':False,
        'method':'deterministic replay of all registered dev seeds; verify payload bytes/hashes, FP16 roundtrip, exact native PCA alias, quality and byte gates',
        'results':replayed,
        'tests':{'passed':2,'failed':0},
        'gate_decision':'FAIL: tiered quality passes, but both byte ratios exceed 0.75; rank-4 alone misses per-task quality and exactly aliases native PCA.'
    }
    (ROOT/'verification_report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__ == '__main__':
    main()
