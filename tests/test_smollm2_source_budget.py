import json
from pathlib import Path


def _budget():
    path=Path(__file__).parents[1]/'experiments'/'transformer'/'T001_smollm2_135m'/'results'/'source_component_budget.json'
    return json.loads(path.read_text())


def test_t001_component_partition_matches_actual_denominator():
    b=_budget(); c=b['components']; n=b['source_denominator_N']
    parts=[
        c['token_embedding_tied_lm_head']['scalars'],
        c['attention_all_30_layers']['scalars'],
        c['mlp_all_30_layers']['scalars'],
        c['two_rmsnorm_scales_per_layer']['scalars'],
        c['final_rmsnorm_scale']['scalars'],
    ]
    assert sum(parts)==n==134_515_008
    assert b['hard_64x_target_bits']==n//4
    assert b['hard_64x_target_bytes']==(n//4)//8


def test_embedding_low_bit_hard_constraints_are_exact_arithmetic():
    b=_budget(); e=b['components']['token_embedding_tied_lm_head']['scalars']; budget=b['hard_64x_target_bits']; h=b['hard_constraints']
    two=h['embedding_at_2_bits_per_scalar']
    assert two['bits']==2*e
    assert two['exceeds_entire_64x_budget_by_bits']==2*e-budget==22_994_352
    assert not two['feasible_even_if_everything_else_cost_zero']
    one=h['embedding_at_1_bit_per_scalar']
    remaining=b['source_denominator_N']-e
    assert one['remaining_bits_for_all_other_scalars']==budget-e==5_317_200
    assert one['remaining_other_scalars']==remaining==106_203_456
    assert abs(one['maximum_average_bits_per_remaining_scalar']-(budget-e)/remaining)<1e-15
