import json
from pathlib import Path
ROOT=Path(__file__).parents[1]
def test_disposition_is_premeasurement_pause():
    p=json.loads((ROOT/'PROTOCOL.json').read_text())
    assert p['measurements_performed'] is False
    assert p['fresh_or_audit_accessed'] is False
    assert 'MA-461' in p['blocker'] and 'MA-462' in p['blocker']
def test_report_labels_limits_and_next_candidate():
    s=(ROOT/'README.md').read_text()
    assert 'NOT ESTABLISHED' in s and 'no measurements' in s
    assert 'MA-596' in (ROOT/'STATUS.md').read_text()
