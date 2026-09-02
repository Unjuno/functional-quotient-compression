import numpy as np
import pytest

from fqc.real_structural_audit import (
    corresponding_weight_family_summary,
    normalized_flat_cosine,
    pairwise_cosine_matrix,
    qk_gqa_group_spans,
    span_summary,
    split_pytorch_linear_input_heads,
    split_pytorch_linear_output_heads,
    swiglu_channel_summary,
    value_output_operators,
)


def test_span_summary_detects_identical_vs_orthogonal_families():
    A=np.eye(2)
    same=span_summary([A,2*A],ranks=(1,2))
    assert same['residual_by_rank']['1'] == pytest.approx(0.0,abs=1e-12)
    X=np.array([[1.,0.],[0.,0.]])
    Y=np.array([[0.,0.],[0.,1.]])
    orth=span_summary([X,Y],ranks=(1,2))
    assert orth['residual_by_rank']['1'] == pytest.approx(0.5)
    assert orth['residual_by_rank']['2'] == pytest.approx(0.0)


def test_head_splits_preserve_pytorch_linear_orientation():
    W=np.arange(24,dtype=float).reshape(6,4)
    outs=split_pytorch_linear_output_heads(W,heads=3,head_dim=2)
    assert np.array_equal(outs[1],W[2:4,:].T)
    O=np.arange(24,dtype=float).reshape(4,6)
    ins=split_pytorch_linear_input_heads(O,heads=3,head_dim=2)
    assert np.array_equal(ins[2],O[:,4:6].T)


def test_value_output_operator_matches_direct_canonical_product():
    rng=np.random.default_rng(1)
    V=rng.normal(size=(4,6))  # 2 kv heads * 2 dim, d_model=6
    O=rng.normal(size=(6,8))  # d_model=6, 4 q heads * 2 dim in PyTorch [out,in]
    ops=value_output_operators(V,O,q_heads=4,kv_heads=2,head_dim=2)
    V0=V[0:2,:].T
    O0=O[:,0:2].T
    O1=O[:,2:4].T
    assert np.allclose(ops[0],V0@O0)
    assert np.allclose(ops[1],V0@O1)


def test_qk_gqa_group_spans_have_expected_membership():
    rng=np.random.default_rng(2)
    Q=rng.normal(size=(8,6)); K=rng.normal(size=(4,6))
    groups=qk_gqa_group_spans(Q,K,q_heads=4,kv_heads=2,head_dim=2)
    assert len(groups)==2
    assert groups[0]['q_heads']==[0,1]
    assert groups[1]['q_heads']==[2,3]
    assert groups[0]['span']['member_count']==3


def test_swiglu_concentration_and_cosine_are_well_formed():
    G=np.eye(4)
    U=np.eye(4)
    D=np.diag([10.,1.,1.,1.])
    r=swiglu_channel_summary(G,U,D)
    assert r['channel_count']==4
    assert r['gate_up_cosine_mean']==pytest.approx(1.0)
    assert r['descriptor_energy_concentration']['top_25_percent_energy_fraction'] > 0.9


def test_corresponding_weight_family_summary_and_cosine_matrix():
    A=np.eye(3); B=-np.eye(3); C=np.eye(3)
    r=corresponding_weight_family_summary({0:A,15:B,29:C})
    assert r['layers']==[0,15,29]
    assert r['span']['residual_by_rank']['1']==pytest.approx(0.0,abs=1e-12)
    assert normalized_flat_cosine(A,B)==pytest.approx(-1.0)
    M=pairwise_cosine_matrix([A,B,C])
    assert M[0][2]==pytest.approx(1.0)
