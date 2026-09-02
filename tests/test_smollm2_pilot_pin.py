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


def test_published_safetensors_size_cross_check_is_exact():
    pin=_pin(); p=pin['config_derived_preflight']; f=pin['checkpoint_file']
    assert f['size_bytes']-p['baseline_bytes_at_16_per_scalar']==p['published_safetensors_minus_scalar_payload_bytes']==30_536
    assert len(f['sha256'])==64


def test_pin_does_not_claim_live_or_compression_evidence():
    pin=_pin()
    assert pin['status']=='PREFLIGHT_ONLY_NO_LIVE_STORAGE_INVENTORY'
    assert pin['authority']['compression_denominator']=='LIVE_STORAGE_INVENTORY_REQUIRED'
    assert pin['authority']['compression_result']=='NOT_TESTED'
