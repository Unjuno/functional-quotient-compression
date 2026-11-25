#!/usr/bin/env python3
"""Verify MA-1059 development-gate fail, provenance and actual serialized state."""
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RUNS=ROOT/'runs'; PROTOCOL=ROOT/'PROTOCOL.json'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=json.loads(PROTOCOL.read_text()); freeze=json.loads((ROOT/'FRESH_FREEZE.json').read_text())
    assert freeze['protocol_sha256']==sha(PROTOCOL)
    assert p['fresh']['locked_before_access'] is True
    dev=json.loads((RUNS/'dev/development.json').read_text()); sel=json.loads((RUNS/'dev/selection.json').read_text())
    assert sel['selected_before_audit_access'] is True and sel['residual_threshold']==0.0
    assert set(sel['selected_updates'])=={'mirror','diag','full','independent'}
    # Threshold zero selects every finite per-series residual by the frozen rule.
    assert len(p['data']['series_ids']['audit'])==24
    payload=json.loads((RUNS/'dev/payloads/manifest.json').read_text())
    for method,entry in payload['methods'].items():
        d=RUNS/'dev/payloads'/method
        n=sum((d/name).stat().st_size for name in entry['files'])
        assert n==entry['adaptation_state_bytes'],(method,n,entry['adaptation_state_bytes'])
        expected=n if method in ('persistence','seasonal_naive') else payload['common_model_bytes']+n
        assert entry['complete_payload_bytes']==expected
    snap=Path('/home/agent/.cache/huggingface/hub/models--amazon--chronos-t5-tiny/snapshots')/p['model']['revision']
    assert sha(snap/'model.safetensors')==freeze['model']['weights_sha256']
    assert sha(Path('/tmp/ma1059-hf/datasets--Monash-University--monash_tsf/snapshots')/p['data']['revision']/p['data']['file'])==p['data']['file_sha256']
    # This is a pre-audit development-gate decision; there are no audit predictions.
    assert not (RUNS/'audit'/'results.json').exists()
    assert dev['selected_updates']==sel['selected_updates']
    assert dev['residual_threshold']==0.0
    manifest=json.loads((RUNS/'input_manifest.json').read_text())
    for rel,h in manifest['development_source_sha256'].items(): assert sha(ROOT/rel)==h
    for rel,h in manifest['development_outputs_sha256'].items(): assert sha(ROOT/rel)==h
    report=json.loads((ROOT/'DEVELOPMENT_DECISION.json').read_text())
    assert report['audit_numeric_values_opened'] is False
    print(json.dumps({'status':'PASS_SCREENING_VERIFICATION','audit_opened':False,'methods_payloads_verified':len(payload['methods']),'complete_payload_bytes':{k:v['complete_payload_bytes'] for k,v in payload['methods'].items()},'selected_residual_threshold':0.0,'residual_allocated_fraction_by_rule':1.0},indent=2))
if __name__=='__main__': main()
