import json
from pathlib import Path

def test_hardware_blocker_is_explicit_and_audit_not_run():
    p=Path(__file__).resolve().parents[1]
    d=json.loads((p/'PROTOCOL.json').read_text())
    assert d['status']=='NOT ESTABLISHED'
    assert d['audit_accessed'] is False
    assert 'zero CUDA devices' in d['hardware_blocker']
