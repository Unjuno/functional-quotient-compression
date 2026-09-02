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
    assert a['status']=='PASS' and a['adapter_version']=='2'
    assert a['serialized_tensor_count']==272
    assert a['serialized_scalar_count_N']==p['expected_unique_scalar_count_N']==134_515_008
    assert a['dtype_scalar_counts']=={'BF16':134_515_008}
    assert a['missing_checkpoint_keys']==0 and a['unaccounted_checkpoint_keys']==0
    assert a['serialized_tensor_payload_bytes']==p['baseline_bytes_at_16_per_scalar']
    assert a['safetensors_container_overhead_bytes']==a['safetensors_prefix_plus_header_bytes']==30_536
    assert f['size_bytes']-a['serialized_tensor_payload_bytes']==30_536
    assert len(f['sha256'])==64


def test_runtime_replay_pass_is_recorded_without_promoting_compression_claims():
    pin=_pin(); authority=pin['authority']; run=pin['actual_runtime_replay']
    assert pin['status']=='RUNTIME_REPLAY_PASS_STRUCTURAL_AUDIT_PENDING'
    assert authority['compression_denominator']=='PINNED_SERIALIZED_CHECKPOINT_UNIQUE_PAID_SCALAR_PAYLOAD'
    assert authority['denominator_N']==134_515_008
    assert authority['runtime_storage_inventory']=='CONSISTENCY_CHECK_PASSED_NOT_DENOMINATOR_AUTHORITY'
    assert run['status']=='PASS'
    assert run['runtime_manifest_unique_scalar_count_N']==authority['denominator_N']
    assert run['embedding_lm_head_same_parameter_object'] and run['embedding_lm_head_same_storage']
    assert run['runtime_source_keys_missing']==0
    assert run['all_replay_cases_passed'] and run['all_replay_cases_max_abs_error']==0.0
    assert run['fresh_runner_repeat_count']==2 and run['fresh_runner_manifest_and_replay_hashes_identical']
    assert authority['structural_compression_evidence']=='NOT_TESTED'
    assert authority['compression_result']=='NOT_TESTED'
