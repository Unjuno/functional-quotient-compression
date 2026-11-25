#!/usr/bin/env python3
"""Post-audit accounting correction for a duplicated pooled-angle artifact.

The generation run wrote the shared pooled angle both inside pooled_basis and
again in each deployed method's one-value code file. Inference needs only one
copy. Remove the redundant common-basis copy and recompute byte totals. No
weights, fitted values, predictions, selections, or audit metrics change.
"""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RUNS=ROOT/'runs'
p=RUNS/'audit/results.json'; result=json.loads(p.read_text())
redundant=RUNS/'audit/payloads/pooled_basis/global_angle.f32le'
if redundant.exists(): redundant.unlink()
basis=RUNS/'audit/payloads/pooled_basis'
basis_bytes=sum(q.stat().st_size for q in basis.iterdir())
for seed in (105803,105804):
 key=f'pooled_lora_seed{seed}'; payload=result['payloads'][key]
 sub=RUNS/'audit/payloads/pooled_lora'/f'seed{seed}'
 code=(sub/'series_codes.f32le').stat().st_size; meta=(sub/'method.json').stat().st_size
 common=payload['common_model_bytes']; payload.update({'adapter_basis_bytes':basis_bytes,'series_code_bytes':code,'metadata_bytes':meta,'complete_payload_bytes':common+basis_bytes+code+meta})
result['accounting_correction']='Removed duplicate serialized pooled global angle: one copy remains in per-seed code; no quality, fit, hyperparameter, or audit data changed.'
p.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result['payloads'].items() if k.startswith('pooled_lora')},indent=2))
