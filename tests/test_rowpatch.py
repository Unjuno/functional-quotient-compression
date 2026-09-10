"""Unit tests for the Gate-4 'rowpatch' section kind (synthetic, no weights)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/t282/code"))

from binary_codec import encode_rowpatch, apply_rowpatch  # noqa: E402


def test_rowpatch_round_trip():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((17, 32)).astype(np.float32)
    d, blob = encode_rowpatch(a, [1, 5, 16])
    assert d["kind"] == "rowpatch" and d["rows"] == [1, 5, 16]
    b = a + 1.0
    out = apply_rowpatch(b, d, blob)
    # rowpatch stores fp16: patched rows equal the fp16-rounded originals
    np.testing.assert_array_equal(out[[1, 5, 16]],
                                  a[[1, 5, 16]].astype("<f2").astype(np.float32))
    np.testing.assert_array_equal(np.delete(out, [1, 5, 16], 0),
                                  np.delete(b, [1, 5, 16], 0))


def test_rowpatch_rejects_unsorted_rows():
    a = np.zeros((8, 8), np.float32)
    with pytest.raises(ValueError):
        encode_rowpatch(a, [3, 1])


def test_rowpatch_rejects_duplicate_rows():
    a = np.zeros((8, 8), np.float32)
    with pytest.raises(ValueError):
        encode_rowpatch(a, [2, 2])


def test_rowpatch_rejects_out_of_range():
    a = np.zeros((8, 8), np.float32)
    with pytest.raises(ValueError):
        encode_rowpatch(a, [8])
    with pytest.raises(ValueError):
        encode_rowpatch(a, [-1])


def test_rowpatch_rejects_bad_extent_and_shape():
    a = np.zeros((8, 8), np.float32)
    d, blob = encode_rowpatch(a, [0, 7])
    with pytest.raises(ValueError):
        apply_rowpatch(a, d, blob[:-2])
    with pytest.raises(ValueError):
        apply_rowpatch(np.zeros((8, 4), np.float32), d, blob)
    bad = dict(d, rows=[0, 9])
    with pytest.raises(ValueError):
        apply_rowpatch(a, bad, blob)


def test_rowpatch_rejects_nonfinite():
    a = np.zeros((8, 8), np.float32)
    d, blob = encode_rowpatch(a, [0])
    n = np.frombuffer(bytearray(blob), dtype="<f2").copy()
    n[0] = np.float16(np.inf)
    with pytest.raises(ValueError):
        apply_rowpatch(a, d, n.tobytes())

def test_sharedmeta_round_trip_and_rate():
    from binary_codec import encode_sharedmeta, decode_sharedmeta
    rng = np.random.default_rng(1)
    a = (rng.standard_normal((16, 128)) * 2).astype(np.float32)
    d, blob = encode_sharedmeta(a, 4, 64, 128)
    assert d["kind"] == "sharedmeta" and d["supergroups"] == 16
    out = decode_sharedmeta(d, blob)
    assert out.shape == a.shape and np.isfinite(out).all()
    # 16 supergroups x (4B meta + 64B codes)
    assert len(blob) == 16 * (4 + 64)
    # coarse check: reconstruction within a few LSB of range/15
    assert float(np.abs(out - a).max()) < float(a.max() - a.min()) / 15 * 3


def test_sharedmeta_rejections():
    from binary_codec import encode_sharedmeta, decode_sharedmeta
    a = np.zeros((4, 128), np.float32)
    with pytest.raises(ValueError):
        encode_sharedmeta(a, 4, 128, 128)
    with pytest.raises(ValueError):
        encode_sharedmeta(a, 4, 64, 64)
    with pytest.raises(ValueError):
        encode_sharedmeta(np.zeros((4, 100), np.float32), 4, 64, 128)
    d, blob = encode_sharedmeta(a + 0.5, 4, 64, 128)
    with pytest.raises(ValueError):
        decode_sharedmeta(d, blob[:-1])
    bad = dict(d, supergroups=3)
    with pytest.raises(ValueError):
        decode_sharedmeta(bad, blob)
