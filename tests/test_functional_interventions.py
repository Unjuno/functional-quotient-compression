import numpy as np
import pytest

from fqc.functional_interventions import (
    bottom_fraction_indices,
    normalized_family_rank_approximation,
    removed_descriptor_energy_fraction,
    swiglu_descriptor_scores,
)


def test_family_rank_approximation_preserves_identical_family_at_rank_one():
    A=np.array([[1.,2.],[3.,4.]])
    r=normalized_family_rank_approximation([A,2*A,-A],1)
    assert r.normalized_family_residual==pytest.approx(0.0,abs=1e-12)
    for got,ref in zip(r.reconstructed,[A,2*A,-A]):
        assert np.allclose(got,ref)


def test_family_rank_residual_matches_orthogonal_reference():
    A=np.array([[1.,0.],[0.,0.]])
    B=np.array([[0.,0.],[0.,1.]])
    r=normalized_family_rank_approximation([A,B],1)
    assert r.normalized_family_residual==pytest.approx(0.5)
    assert len(r.per_member_relative_frobenius_change)==2


def test_bottom_fraction_is_stable_and_nonempty():
    scores=np.array([3.,1.,1.,2.,5.])
    idx=bottom_fraction_indices(scores,0.4)
    assert idx.tolist()==[1,2]
    assert bottom_fraction_indices(scores,0.01).size==1


def test_swiglu_descriptor_scores_and_removed_energy():
    U=np.array([[3.,4.],[1.,0.],[0.,2.]])
    D=np.array([[2.,0.,0.],[0.,3.,4.]])
    scores=swiglu_descriptor_scores(U,D)
    assert np.allclose(scores,[10.,3.,8.])
    frac=removed_descriptor_energy_fraction(scores,[1])
    assert frac==pytest.approx(9/(100+9+64))


def test_invalid_fraction_and_shapes_fail_closed():
    with pytest.raises(ValueError): bottom_fraction_indices([1,2,3],1.0)
    with pytest.raises(ValueError): normalized_family_rank_approximation([np.eye(2)],2)
    with pytest.raises(ValueError): swiglu_descriptor_scores(np.ones((2,3)),np.ones((2,2)))
