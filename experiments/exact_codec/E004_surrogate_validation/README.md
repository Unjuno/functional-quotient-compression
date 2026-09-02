# E004 — Surrogate ranking with exact validation

Status: **commit rule reconstructed**.

The archived tree-rotation study found only modest agreement between a cheap
surrogate and the true hard-budget objective. The durable algorithmic lesson is
therefore not "Top-3 is always enough". It is:

1. use a surrogate to rank expensive candidates;
2. evaluate a small prefix with the exact hard-budget objective;
3. commit only a validated exact improvement.

`src/fqc/validation.py` encodes that rule. The archived Top-3 numerical results
remain empirical provenance rather than a theorem or default constant.
