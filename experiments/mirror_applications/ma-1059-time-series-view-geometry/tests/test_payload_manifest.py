import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

METHODS={'native','pooled_lora','mirror','diag','full','independent','mirror_trend','persistence','seasonal_naive'}

def test_serialized_dev_payload_bytes_are_exact():
    m=json.loads((ROOT/'runs/dev/payloads/manifest.json').read_text())
    assert set(m['methods'])==METHODS
    for name,entry in m['methods'].items():
        d=ROOT/'runs/dev/payloads'/name
        assert sum((d/f).stat().st_size for f in entry['files'])==entry['adaptation_state_bytes']
        expected=entry['adaptation_state_bytes'] if name in ('persistence','seasonal_naive') else m['common_model_bytes']+entry['adaptation_state_bytes']
        assert expected==entry['complete_payload_bytes']

def test_fresh_serialized_payloads_and_forecast_artifacts_are_complete():
    p=json.loads((ROOT/'PROTOCOL.json').read_text())
    freeze=json.loads((ROOT/'AUDIT_FREEZE.json').read_text())
    results=json.loads((ROOT/'runs/audit/results.json').read_text())
    replay=json.loads((ROOT/'runs/audit/verification_replay.json').read_text())
    assert freeze['audit_numeric_values_opened'] is False
    assert freeze['selection_changes_after_audit_forbidden'] is True
    assert results['audit_series_ids']==p['fresh']['audit_ids']
    assert results['seeds']==p['fresh']['sampling_seeds']
    assert set(results['methods']['105903'])==METHODS
    model_bytes=json.loads((ROOT/'runs/audit/payloads/common_model_manifest.json').read_text())['complete_model_bytes']
    for name,entry in results['payloads'].items():
        d=ROOT/'runs/audit/payloads'
        assert sum((d/f).stat().st_size for f in entry['files'])==entry['method_state_bytes']
        expected=entry['method_state_bytes'] if name in ('persistence','seasonal_naive') else model_bytes+entry['method_state_bytes']
        assert expected==entry['complete_payload_bytes']
    for seed in results['seeds']:
        for method in METHODS:
            assert (ROOT/'runs/audit'/f'{method}_seed{seed}.npz').exists()
    assert replay['status']=='PASS'
    assert replay['serialized_payload_forecast_exact'] is True
    assert replay['max_metric_difference']==0
