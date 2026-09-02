import numpy as np
import pytest

from fqc.continuation_metrics import pearson_correlation, relative_rms_delta, safe_ratio


def test_relative_rms_delta_and_mask():
    r=np.array([[[1.,2.],[3.,4.]],[[5.,6.],[7.,8.]]])
    p=r.copy(); p[0,0,0]+=1
    mask=np.array([[True,False],[False,False]])
    got=relative_rms_delta(r,p,mask)
    expected=np.sqrt(1/2)/np.sqrt((1+4)/2)
    assert got==pytest.approx(expected)


def test_relative_rms_rejects_shape_and_empty_mask():
    with pytest.raises(ValueError): relative_rms_delta(np.zeros((2,3)),np.zeros((3,2)))
    with pytest.raises(ValueError): relative_rms_delta(np.zeros((1,2,3)),np.zeros((1,2,3)),np.zeros((1,2),dtype=bool))


def test_safe_ratio_and_pearson():
    assert safe_ratio(2.0,4.0)==pytest.approx(0.5)
    assert pearson_correlation([1,2,3],[2,4,6])==pytest.approx(1.0)
    assert pearson_correlation([1,2,3],[6,4,2])==pytest.approx(-1.0)
