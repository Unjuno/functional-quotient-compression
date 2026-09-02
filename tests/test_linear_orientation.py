import numpy as np
import pytest

from fqc.linear_orientation import (
    PYTORCH_OUT_IN, ROW_VECTOR_X_W,
    canonical_row_matrix, expected_attention_projection_shapes,
)


def test_attention_shapes_for_pytorch_and_canonical_orientation_are_transposes():
    row=expected_attention_projection_shapes(ROW_VECTOR_X_W,576,9,3,64,64)
    pt=expected_attention_projection_shapes(PYTORCH_OUT_IN,576,9,3,64,64)
    assert row['WQ']==[576,576] and pt['WQ']==[576,576]
    assert row['WK']==[576,192] and pt['WK']==[192,576]
    assert row['WV']==[576,192] and pt['WV']==[192,576]
    assert row['WO']==[576,576] and pt['WO']==[576,576]


def test_pytorch_weight_is_transposed_into_row_operator():
    raw=np.arange(6).reshape(2,3)
    assert np.array_equal(canonical_row_matrix(raw,PYTORCH_OUT_IN),raw.T)
    assert np.array_equal(canonical_row_matrix(raw,ROW_VECTOR_X_W),raw)


def test_unknown_orientation_fails_closed_for_conversion():
    with pytest.raises(ValueError): canonical_row_matrix(np.zeros((2,2)),'mystery')
