"""Recheck MA-880 pair isolation, corrected replay identity, payloads and gates."""
import importlib.util
import json
import hashlib
import statistics
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'source' / 'experiment.py'
spec = importlib.util.spec_from_file_location('ma880_verify', SRC)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

result = json.loads((ROOT / 'fresh_results.json').read_text())
raw = json.loads((ROOT / 'fresh_results_unrotated_deviation.json').read_text())
freeze = json.loads((ROOT / 'FROZEN_SETTINGS.json').read_text())
assert result['seeds'] == [983, 997, 1009]
assert result['q'] == freeze['selected_cp_rank'] == 1
assert result['steps'] == freeze['updates'] == 600
assert freeze['fresh_corrected_opened'] is False
assert [list(p) for p in module.TRAIN_PAIRS] == freeze['train_pairs']
assert [list(p) for p in module.FRESH_PAIRS] == freeze['fresh_pairs']

# Corrected replay uses the same predeclared row IDs as the unrotated implementation-deviation run.
for new, old in zip(result['rows'], raw['rows']):
    assert new['seed'] == old['seed']
    assert new['prompt_rows'] == old['prompt_rows']
    assert new['factor_payload_bytes'] == 583915
    assert new['independent_full_payload_bytes'] == 4594139
    payload = ROOT.parent / new['factor_payload_path']
    data = payload.read_bytes()
    assert len(data) == new['factor_payload_bytes']
    assert hashlib.sha256(data).hexdigest() == new['factor_payload_sha256']
    assert new['factor_bytes_saved_fraction'] >= 0.50
    for pair_result in new['pair_results']:
        metrics = pair_result['metrics']
        assert len(pair_result['prompt_rows']) == 12
        assert metrics['factor']['mean_nll'] - metrics['full']['mean_nll'] > 0.50
        assert metrics['factor']['delta_native'] > 1.00

# Confirm tokenizer-aligned prompt identity and pairwise disjoint fresh chunks.
tokenizers = {}
for model_id in module.MODELS:
    snap = module.snapshot(model_id)
    tokenizers[model_id] = AutoTokenizer.from_pretrained(snap, local_files_only=True, use_fast=True)
for pair in module.FRESH_PAIRS:
    cal_rows, test_rows, cal_texts, test_texts = set(), set(), set(), set()
    for seed, row in zip(result['seeds'], result['rows']):
        key = f'{pair[0]}>{pair[1]}'
        cal = module.select_prompts('validation', pair, tokenizers, seed, 12, 'cal')
        test = module.select_prompts('test', pair, tokenizers, seed, 12, 'test')
        assert [x['row'] for x in cal] == row['prompt_rows']['heldout_calibration'][key]
        assert [x['row'] for x in test] == row['prompt_rows']['heldout_test'][key]
        assert not cal_rows.intersection(x['row'] for x in cal)
        assert not test_rows.intersection(x['row'] for x in test)
        cal_rows.update(x['row'] for x in cal); test_rows.update(x['row'] for x in test)
        cal_texts.update(x['text'] for x in cal); test_texts.update(x['text'] for x in test)
    assert len(cal_rows) == 36 and len(test_rows) == 36
    assert not cal_texts.intersection(test_texts)

print('MA-880 verification PASS: frozen train-only Q, 2 whole ordered fresh pairs × 3 seeds, exact corrected/unrotated prompt manifest match, 36 disjoint calibration/test rows per pair, payload length/hash and quality/storage gates replayed.')
