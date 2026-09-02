import numpy as np
import pytest
from fqc.structural_diagnostics import (
    operator_span_spectrum, normalized_commutator, symmetric_orthogonal_jd_eligible,
    fixed_support_residual, gqa_consumer_multiplicity, channel_outer_descriptor,
)


def test_commuting_can_have_disjoint_support_d58_counterexample():
    X0=np.diag([1.0,0.0]); X1=np.diag([0.0,2.0]); Q=np.eye(2)
    assert normalized_commutator(X0,X1)==0.0
    assert fixed_support_residual(X0,Q,[0])[1]==0.0
    assert fixed_support_residual(X1,Q,[0])[1]==1.0
    assert fixed_support_residual(X0,Q,[0,1])[1]==0.0
    assert fixed_support_residual(X1,Q,[0,1])[1]==0.0


def test_common_support_does_not_imply_commutation():
    X=np.array([[1.,0.],[0.,0.]])
    Y=np.array([[0.,1.],[1.,0.]])
    assert fixed_support_residual(X,np.eye(2),[0,1])[1]==0.0
    assert fixed_support_residual(Y,np.eye(2),[0,1])[1]==0.0
    assert normalized_commutator(X,Y)>0


def test_low_operator_span_does_not_imply_commutation():
    A=np.array([[1.,0.],[0.,-1.]])
    B=np.array([[0.,1.],[1.,0.]])
    C=2*A-0.5*B
    _,res=operator_span_spectrum([A,B,C])
    assert np.linalg.matrix_rank(np.stack([A.ravel(),B.ravel(),C.ravel()]),tol=1e-10)==2
    assert res[2]<1e-28
    assert normalized_commutator(A,B)>0


def test_noisy_three_generator_dictionary_has_tiny_rank3_residual():
    rng=np.random.default_rng(7)
    G=rng.normal(size=(3,4,4)); atoms=[]
    for _ in range(24):
        c=rng.normal(size=3); atoms.append(np.tensordot(c,G,axes=1)+1e-4*rng.normal(size=(4,4)))
    _,res=operator_span_spectrum(atoms)
    assert res[3]<1e-6


def test_gqa_reuse_and_swiglu_scaling_invariant():
    mapping=[i//4 for i in range(32)]
    assert gqa_consumer_multiplicity(mapping,8)==(4,)*8
    z=np.array([1.,-2.,0.5]); d=np.array([.7,-.1]); c=3.7
    assert np.allclose(channel_outer_descriptor(z,d),channel_outer_descriptor(c*z,d/c))


def test_symmetric_theorem_eligibility_is_not_generic_commutator_api():
    A=np.array([[1.,1.],[0.,1.]])
    assert not symmetric_orthogonal_jd_eligible([A])
    assert symmetric_orthogonal_jd_eligible([np.diag([1.,2.])])
    with pytest.raises(ValueError): fixed_support_residual(np.eye(2),np.array([[1.,1.],[0.,1.]]),[0])
