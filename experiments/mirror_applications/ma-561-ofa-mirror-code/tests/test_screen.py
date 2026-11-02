from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from source.verify import verify

def test_fresh_replay_and_actual_payloads():
    result=verify(ROOT)
    assert result['fresh_worlds']==3
    assert result['metric_rows_checked']==48

def test_protocol_locked_before_fresh():
    import json
    p=json.loads((ROOT/'PROTOCOL.json').read_text())
    a=json.loads((ROOT/'AMENDMENT_A1.json').read_text())
    assert p['fresh']['locked_before_access'] is True
    assert a['fresh_accessed_before_amendment'] is False
