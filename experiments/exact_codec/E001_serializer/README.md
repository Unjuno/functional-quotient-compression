# E001 — Serializer-aware hard-budget reconstruction

Status: **partial canonical reconstruction**.

The handoff records an original 4-leaf experiment with 152 legal states and a
serializer

```text
8 * ceil((B_raw + 5) / 8)
```

that changes the hard-budget optimum. In particular, a 76-logical-bit state
serializes to 88 bits and therefore must fail an 80-bit budget.

The original full 152-state generator was not present in the handoff packages.
The canonical repository therefore separates two claims:

1. **Recovered exactly:** the serializer rule and the 76 -> 88 false-pass
   counterexample.
2. **Not yet recovered:** the original 152-state dataset and the reported
   23.5505-at-72-bit optimum.

Tests in `tests/test_serializer.py` and `tests/test_pareto.py` lock the recovered
mechanism without pretending to reproduce the missing original state universe.
