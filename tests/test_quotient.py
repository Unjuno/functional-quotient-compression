import pytest

from fqc.quotient import QuotientCandidate, QuotientKey, quotient_candidates


def test_same_decoded_and_prerequisites_merge_to_cheapest_description() -> None:
    key = QuotientKey(decoded=(1, 0, 1), prerequisites=("root-a",))
    out = quotient_candidates(
        [
            QuotientCandidate(key, bits=19, distortion=3.5, payload="path-a"),
            QuotientCandidate(key, bits=13, distortion=3.5, payload="path-b"),
        ]
    )
    assert len(out) == 1
    assert out[0].bits == 13


def test_same_decoded_mask_different_prerequisites_do_not_merge() -> None:
    candidates = [
        QuotientCandidate(QuotientKey((1, 0), ("root-a",)), 10, 2.0),
        QuotientCandidate(QuotientKey((1, 0), ("root-b",)), 9, 2.0),
    ]
    assert len(quotient_candidates(candidates)) == 2


def test_distortion_mismatch_rejects_invalid_equivalence_class() -> None:
    key = QuotientKey(decoded=(1, 1), prerequisites=())
    with pytest.raises(ValueError):
        quotient_candidates(
            [
                QuotientCandidate(key, 8, 1.0),
                QuotientCandidate(key, 7, 1.1),
            ]
        )
