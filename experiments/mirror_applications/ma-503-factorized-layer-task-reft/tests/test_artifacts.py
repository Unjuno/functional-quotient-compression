import csv
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
def test_results_and_serialized_payloads():
    rows=list(csv.DictReader((ROOT/'RESULTS_CORE.csv').open()))
    assert len(rows)==108
    assert {r['phase'] for r in rows}=={'fresh'}
    for r in rows:
        p=ROOT/r['payload_path']
        assert p.exists() and p.stat().st_size==int(r['serialized_payload_bytes'])
        assert float(r['max_serialized_replay_abs_error'])==0.0
