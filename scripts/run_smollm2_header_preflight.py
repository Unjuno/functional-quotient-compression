#!/usr/bin/env python3
"""Run T001 SmolLM2 serialized-checkpoint header preflight."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fqc.hf_llama_adapter import build_hf_llama_adapter_plan
from fqc.manifest_builder import sha256_json
from fqc.safetensors_inventory import (
    materialize_adapter_plan_from_safetensors_header,
    read_safetensors_header,
    sha256_file,
)


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--pin',default='experiments/transformer/T001_smollm2_135m/pilot_pin.json')
    ap.add_argument('--output-dir',required=True)
    args=ap.parse_args()
    checkpoint=Path(args.checkpoint); pin=json.loads(Path(args.pin).read_text())
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)

    expected_file=pin['checkpoint_file']
    actual_sha=sha256_file(checkpoint); actual_size=checkpoint.stat().st_size
    if actual_sha!=expected_file['sha256']:
        raise SystemExit(f'checkpoint SHA256 mismatch: {actual_sha}')
    if actual_size!=expected_file['size_bytes']:
        raise SystemExit(f'checkpoint size mismatch: {actual_size}')

    header=read_safetensors_header(checkpoint)
    plan=build_hf_llama_adapter_plan(
        pin['config'],checkpoint_sha256='sha256:'+actual_sha,model_id=pin['model_id'],
        available_checkpoint_keys=header.entries,
    )
    preflight=materialize_adapter_plan_from_safetensors_header(plan,header)
    manifest=dict(preflight.extraction_manifest)
    manifest['model_identity']['hub_revision']=pin['hub_revision']
    manifest['model_identity']['checkpoint_file']=expected_file['path']
    manifest['model_identity']['checkpoint_file_size_bytes']=actual_size
    manifest_hash=sha256_json(manifest)
    n=manifest['model_identity']['unique_baseline_scalar_count_N']
    result={
        'experiment_id':pin['experiment_id'],
        'status':'PASS' if preflight.passed else 'FAIL',
        'hub_revision':pin['hub_revision'],
        'checkpoint_sha256':actual_sha,
        'checkpoint_size_bytes':actual_size,
        'safetensors_header_size_bytes':header.header_size_bytes,
        'serialized_tensor_count':preflight.serialized_tensor_count,
        'serialized_scalar_count':preflight.serialized_scalar_count,
        'dtype_scalar_counts':dict(preflight.dtype_scalar_counts),
        'config_expected_scalar_count':preflight.config_expected_scalar_count,
        'live_D57_unique_baseline_scalar_count_N':n,
        'hard_64x_target_bits':n//4,
        'hard_64x_target_bytes':(n//4)//8,
        'unaccounted_checkpoint_keys':list(preflight.unaccounted_checkpoint_keys),
        'missing_checkpoint_keys':list(preflight.missing_checkpoint_keys),
        'extraction_manifest_sha256':manifest_hash,
        'evidence_scope':'SERIALIZED_CHECKPOINT_HEADER_ONLY_NO_MODULE_REPLAY',
    }
    pin_pre=pin['config_derived_preflight']
    required=(
        preflight.passed and
        n==pin_pre['expected_unique_scalar_count_N']==preflight.serialized_scalar_count and
        n//4==pin_pre['hard_64x_target_bits'] and
        preflight.serialized_tensor_count==272 and
        preflight.dtype_scalar_counts=={'BF16':n}
    )
    result['status']='PASS' if required else 'FAIL'
    (out/'preflight_result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (out/'extraction_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0 if required else 1


if __name__=='__main__':
    raise SystemExit(main())
