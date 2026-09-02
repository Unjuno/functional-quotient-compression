# E003 — Decoded-state quotient

Status: **equivalence contract reconstructed**.

The handoff reports 2,090,918 recursive optimizer states for a 16-leaf toy but
only 65,536 decoded binary masks, suggesting about 31.9x state redundancy.

A critical condition is now explicit in code: decoded output alone is not a
safe quotient key when two paths leave different decoder prerequisites or future
capabilities. Canonical merging therefore uses

```text
(decoded_signature, prerequisite_signature)
```

and rejects a quotient class if supposedly equivalent states have different
distortion. The archived 31.9x count is retained as exact-toy evidence, while
the canonical implementation encodes the stronger safety condition needed for
a general optimizer.
