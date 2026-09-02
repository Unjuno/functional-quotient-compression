# E005 — Description-only rotation transfer

Status: **identity mechanism reconstructed**.

For a description-only local tree rotation, the handoff records

```text
C_rotated(M) = C_base(M) + delta(M ∩ R)
```

and exact transfer through a conditional frontier when decoded distortion is
layout-independent.

`src/fqc/transfer.py` implements the corresponding conditional-budget transfer,
and `tests/test_transfer.py` verifies equality of the optimum distortion value against direct enumeration across all
256 masks of an independent toy system and multiple hard budgets. Tied optimal masks
need not be identical because tie-breaking is outside the transfer theorem.

This cache reuse is invalid for representation-changing operations such as
precision, root, or training updates unless their distortion statistics are
recomputed.
