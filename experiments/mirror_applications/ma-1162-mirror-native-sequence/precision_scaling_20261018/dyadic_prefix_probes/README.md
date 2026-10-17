# MA-1162 — Dyadic scalar coordinate / progressive precision numerical probe

This is an **exploratory numerical diagnostic**, not a trained NN, prospective fresh/audit evaluation, or claim of new independent model capacity. The original precision-scaling study is unchanged.

**Mirror insertion:** One progressively precise dyadic scalar is the sole proposed physical learning variable, accessed by four fixed orthogonal Mirror + tanh + inverse-Mirror residual steps. No role angles or gains are independently learned.

Goals: check exact one-ULP fixed-point perturbations against binary64 conversion, nested prefix continuity, and the local response of a repeated nonlinear Mirror operator.

Reproduction from this directory:

```bash
PYTHONPATH=source python -m pytest -q tests/test_precision_coordinate_probe.py
python source/precision_coordinate_probe.py
```

The script writes `source/results.csv`; compare its numeric columns with the committed `results.csv` (wall times vary by hardware). Requirements: Python, mpmath, pytest. No optimizer, images, or GPU required.

## Findings

- 4 tests PASS; 7 fractional precision conditions from 16 to 800 bits.
- Dyadic spacing and parent-child prefix nesting hold exactly.
- In the tested fixture, ordinary binary64 rounds adjacent dyadic values to the same number from p=64 onwards, while arbitrary-precision evaluation distinguishes their nonlinear Mirror outputs.
- Local output sensitivity remains approximately 4.084; finer stored bits alone do **not** demonstrate new independently learnable roles.
- The previous 8,000-bit experiment already stored its scalar in `N/2^p` fixed-point form, while computing Mirror functions via arbitrary-precision floating-point arithmetic.

See [Japanese report](REPORT_JA.md), [CSV](results.csv), [protocol](PROTOCOL_EXPLORATORY.json), and [verification](VERIFICATION.json).
