# MA-1162 — Dyadic scalar coordinate / progressive precision numerical probe

This is an **exploratory numeric experiment**, not a trained NN, not a prospective fresh/audit evaluation, and not a claim of independent capacity. Original precision scaling study is unchanged.

**Mirror insertion:** use one progressively precise dyadic scalar as the only proposed physical trainable variable, accessed by four fixed orthogonal Mirror + tanh + inverse-Mirror residual steps. No role gains or angles are independently learned.

Goals: compare one-ULP fixed-point perturbations versus binary64 conversion collapse, confirm nested prefix continuity as extra fractional bits are added, and measure local nonlinear Mirror response.

Reproduction (from this directory):

`PYTHONPATH=source python -m pytest -q tests/test_precision_coordinate_probe.py
python source/precision_coordinate_probe.py`

The script writes a new `source/results.csv` (same data as committed `results.csv` for its numeric columns, except wall time may vary with hardware). It does not train a model. Dependencies: Python, `mpmath`, `pytest`. `PROTOCOL_EXPLORATORY.json` and `VERIFICATION.json` describe limitations.

## Outcomes

- 4 local tests PASS; 7 precision conditions (fractional bits 16–800).
- Dyadic spacing and parent-child prefix nesting hold exactly.
- For this scalar/curve, ordinary binary64 converts `s_p` and `s_p+2^-p` to the same number for p>=64, while the arbitrary-precision recurrence distinguishes them.
- Output sensitivity remains near 4.084, so finer stored bits produce proportionally smaller output changes, not demonstrably new independently learnable functionality.
- The earlier 8,000-bit precision study already used `N/2^p` fixed-point storage, despite using arbitrary precision floating arithmetic for the Mirror operations.

See [Japanese report](REPORT_JA.md) and [CSV](results.csv).
