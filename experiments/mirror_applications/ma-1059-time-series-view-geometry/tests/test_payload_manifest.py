import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_serialized_dev_payload_bytes_are_exact_and_audit_is_unopened():
    m=json.loads((ROOT/'runs/dev/payloads/manifest.json').read_text())
    assert set(m['methods'])=={'native','pooled_lora','mirror','diag','full','independent','mirror_trend','persistence','seasonal_naive'}
    for name,entry in m['methods'].items():
        d=ROOT/'runs/dev/payloads'/name
        assert sum((d/f).stat().st_size for f in entry['files'])==entry['adaptation_state_bytes']
        expected=entry['adaptation_state_bytes'] if name in ('persistence','seasonal_naive') else m['common_model_bytes']+entry['adaptation_state_bytes']
        assert expected==entry['complete_payload_bytes']
    p=json.loads((ROOT/'PROTOCOL.json').read_text())
    assert len(p['data']['series_ids']['audit'])==24
    assert not (ROOT/'runs/audit/results.json').exists()
