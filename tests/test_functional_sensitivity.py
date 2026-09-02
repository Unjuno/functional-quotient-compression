import numpy as np
from fqc.functional_sensitivity import (
    softmax_rows, bilinear_logit_bound, softmax_half_lipschitz_bound,
    attention_output_bound, attention_output_bound_coarse,
    top1_preservation_certificate, cross_entropy, cross_entropy_change_bounds,
    telescoping_composition_bound,
)


def test_bilinear_logit_and_softmax_bounds():
    rng=np.random.default_rng(11)
    X=rng.normal(size=(5,4)); dP=.03*rng.normal(size=(4,4)); alpha=.7
    dL=alpha*X@dP@X.T
    assert np.linalg.norm(dL,'fro') <= bilinear_logit_bound(X,dP,alpha)+1e-12
    L=rng.normal(size=(5,5)); A=softmax_rows(L); Ap=softmax_rows(L+dL)
    assert np.linalg.norm(Ap-A,'fro') <= softmax_half_lipschitz_bound(dL)+1e-12


def test_attention_output_bound_and_coarse_order():
    rng=np.random.default_rng(12)
    X=rng.normal(size=(5,4)); R=rng.normal(size=(4,4)); dR=.02*rng.normal(size=(4,4))
    L=rng.normal(size=(5,5)); Lp=L+.03*rng.normal(size=(5,5))
    Y=softmax_rows(L)@X@R; Yp=softmax_rows(Lp)@X@(R+dR)
    exact=np.linalg.norm(Yp-Y,'fro')
    fine=attention_output_bound(X,R,dR,L,Lp); coarse=attention_output_bound_coarse(X,R,dR,L,Lp)
    assert exact <= fine+1e-12
    assert fine <= coarse+1e-12


def test_final_logit_margin_and_cross_entropy_bounds_match_d61_logic():
    z=np.array([2.8,1.9,.4,-.7]); dz=np.array([-.08,.04,.03,.01]); zp=z+dz
    assert top1_preservation_certificate(z,.08)
    actual=abs(cross_entropy(zp,0)-cross_entropy(z,0))
    linf,l2=cross_entropy_change_bounds(dz)
    assert actual <= linf+1e-12 and actual <= l2+1e-12
    assert abs(linf-.16)<1e-12


def test_two_stage_telescope_reproduces_d61_0186():
    assert abs(telescoping_composition_bound([.08,.05],[1.0,1.7])-.186)<1e-12
