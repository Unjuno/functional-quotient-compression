from __future__ import annotations


def byte_aligned_bits(raw_bits: int, *, framing_bits: int = 5, byte_bits: int = 8) -> int:
    """Return serialized length for a framed, byte-aligned payload.

    This is the serializer model recorded in the E1 handoff:
        byte_bits * ceil((raw_bits + framing_bits) / byte_bits)

    Parameters
    ----------
    raw_bits:
        Logical payload bits before framing/alignment.
    framing_bits:
        Serializer overhead added before alignment.
    byte_bits:
        Alignment unit. Eight corresponds to ordinary bytes.
    """
    if raw_bits < 0:
        raise ValueError("raw_bits must be non-negative")
    if framing_bits < 0:
        raise ValueError("framing_bits must be non-negative")
    if byte_bits <= 0:
        raise ValueError("byte_bits must be positive")
    total = raw_bits + framing_bits
    return byte_bits * ((total + byte_bits - 1) // byte_bits)
