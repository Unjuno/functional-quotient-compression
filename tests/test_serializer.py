from fqc.serializer import byte_aligned_bits


def test_e1_reported_raw_76_serializes_to_88_bits() -> None:
    assert byte_aligned_bits(76, framing_bits=5, byte_bits=8) == 88


def test_raw_budget_can_false_pass() -> None:
    raw_budget = 80
    raw_bits = 76
    assert raw_bits <= raw_budget
    assert byte_aligned_bits(raw_bits) > raw_budget
