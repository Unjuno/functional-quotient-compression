import numpy as np
import pytest

from fqc.replay import ReplayCase, build_replay_witness, compare_replay_case

CONTRACT={'contract_id':'module-replay-v1','atol':1e-6,'rtol':1e-5,'evaluation_dtype':'float64'}


def test_replay_passes_within_predeclared_tolerance():
    c=ReplayCase('a',{'x':np.array([1.,2.])},np.array([2.,4.]),np.array([2.000001,3.999999]))
    r=compare_replay_case(c,CONTRACT)
    assert r.passed
    assert r.input_hashes['x'].startswith('sha256:')
    assert r.max_tolerance_ratio<=1.0


def test_replay_fails_outside_tolerance():
    c=ReplayCase('a',{},np.array([1.]),np.array([1.1]))
    r=compare_replay_case(c,CONTRACT)
    assert not r.passed and r.reason=='tolerance exceeded'
    assert r.max_tolerance_ratio>1.0


def test_shape_and_nonfinite_outputs_fail_closed():
    shape=compare_replay_case(ReplayCase('s',{},np.zeros(2),np.zeros(3)),CONTRACT)
    bad=compare_replay_case(ReplayCase('n',{},np.array([1.]),np.array([np.nan])),CONTRACT)
    assert not shape.passed and shape.reason=='shape mismatch'
    assert not bad.passed and bad.reason=='non-finite output'


def test_witness_is_order_independent_and_hashes_contract():
    a=ReplayCase('a',{'x':np.array([1])},np.array([1.]),np.array([1.]))
    b=ReplayCase('b',{'x':np.array([2])},np.array([2.]),np.array([2.]))
    x=build_replay_witness([a,b],CONTRACT); y=build_replay_witness([b,a],CONTRACT)
    assert x==y and x.passed and x.contract_hash.startswith('sha256:')


def test_invalid_tolerance_contract_is_rejected():
    with pytest.raises(ValueError):
        build_replay_witness([],{'atol':-1,'rtol':0})
