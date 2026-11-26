#!/usr/bin/env python3
"""Verify MA-1059 corrected development outputs, split and actual serialized payloads."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RUNS=ROOT/'runs'; PROTOCOL=ROOT/'PROTOCOL.json'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=json.loads(PROTOCOL.read_text()); freeze=json.loads((ROOT/'FRESH_FREEZE.json').read_text())
    assert freeze['protocol_sha256']==sha(PROTOCOL) and freeze['audit_numeric_values_opened'] is False
    assert p['fresh']['locked_before_access'] is True
    assert p['amendments'][0]['id']=='AM-01'
    dev=json.loads((RUNS/'dev/development.json').read_text()); sel=json.loads((RUNS/'dev/selection.json').read_text())
    assert sel['selected_before_audit_access'] is True
    assert sel['residual_min_relative_gain']==dev['residual_min_relative_gain']
    assert sel['development_residual_allocation_fraction']<=.25
    assert set(sel['selected_updates'])=={'mirror','diag','full','independent'}
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
    data_path=Path('/tmp/ma1059-hf/datasets--Monash-University--monash_tsf/snapshots')/p['data']['revision']/p['data']['file']
    assert sha(data_path)==p['data']['file_sha256']
    assert not (RUNS/'audit'/'results.json').exists()
    assert dev['selected_updates']==sel['selected_updates']
    manifest=json.loads((RUNS/'input_manifest.json').read_text())
    for rel,h in manifest['development_source_sha256'].items(): assert sha(ROOT/rel)==h
    for rel,h in manifest['development_outputs_sha256'].items(): assert sha(ROOT/rel)==h
    decision=json.loads((ROOT/'DEVELOPMENT_DECISION.json').read_text())
    assert decision['audit_numeric_values_opened'] is False
    print(json.dumps({'status':'PASS_CORRECTED_DEV_VERIFICATION','audit_opened':False,'methods_payloads_verified':len(payload['methods']),'residual_min_relative_gain':sel['residual_min_relative_gain'],'dev_residual_fraction':sel['development_residual_allocation_fraction'],'complete_payload_bytes':{k:v['complete_payload_bytes'] for k,v in payload['methods'].items()}},indent=2))
if __name__=='__main__':main()
