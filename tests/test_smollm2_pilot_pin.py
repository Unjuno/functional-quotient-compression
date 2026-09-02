import json
from pathlib import Path

from fqc.hf_llama_adapter import expected_hf_llama_unique_scalars


def _pin():
    path=Path(__file__).parents[1]/'experiments'/'transformer'/'T001_smollm2_135m'/'pilot_pin.json'
    return json.loads(path.read_text())


def test_smollm2_pin_config_and_hard_budget_are_internally_consistent():
    pin=_pin(); cfg=pin['config']; p=pin['config_derived_preflight']
    n=expected_hf_llama_unique_scalars(cfg)
    assert n==134_515_008==p['expected_unique_scalar_count_N']
    assert p['baseline_bits_at_16_per_scalar']==16*n
    assert p['baseline_bytes_at_16_per_scalar']==2*n
    assert p['hard_64x_target_bits']==n//4
    assert p['hard_64x_target_bytes']==(n//4)//8


def test_actual_serialized_checkpoint_matches_preflight_cross_checks():
    pin=_pin(); p=pin['config_derived_preflight']; f=pin['checkpoint_file']; a=pin['actual_serialized_checkpoint']
    assert a['status']=='PASS'
    assert a['serialized_tensor_count']==272
    assert a['serialized_scalar_count_N']==p['expected_unique_scalar_count_N']==134_515_008
    assert a['dtype_scalar_counts']=={'BF16':134_515_008}
    assert a['missing_checkpoint_keys']==0 and a['unaccounted_checkpoint_keys']==0
    assert a['serialized_tensor_payload_bytes']==p['baseline_bytes_at_16_per_scalar']
    assert a['safetensors_container_overhead_bytes']==a['safetensors_prefix_plus_header_bytes']==30_536
    assert f['size_bytes']-a['serialized_tensor_payload_bytes']==30_536
    assert len(f['sha256'])==64


def test_pin_promotes_source_denominator_without_claiming_runtime_or_compression_evidence():
    pin=_pin(); authority=pin['authority']
    assert pin['status']=='SERIALIZED_CHECKPOINT_HEADER_PASS_RUNTIME_REPLAY_PENDING'
    assert authority['compression_denominator']=='PINNED_SERIALIZED_CHECKPOINT_UNIQUE_PAID_SCALAR_PAYLOAD'
    assert authority['denominator_N']==134_515_008
    assert authority['runtime_storage_inventory']=='CONSISTENCY_CHECK_REQUIRED_NOT_DENOMINATOR_AUTHORITY'
    assert authority['module_replay']=='REQUIRED_BEFORE_STRUCTURAL_COMPRESSION_CLAIMS'
    assert authority['compression_result']=='NOT_TESTED'
