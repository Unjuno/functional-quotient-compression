# Module Replay Witness

A valid D57 manifest can still encode the wrong architecture semantics. The
replay witness checks that outputs reconstructed from extracted primitives match
outputs from the original module on the same declared inputs.

## Contract first

Numerical tolerances are part of the replay contract and are hashed before the
results are interpreted. They must not be loosened after looking at a failing
case without creating a new contract/version.

The current witness records, per replay case:

- hashes of every declared input;
- reference-output hash;
- extracted-output hash;
- maximum absolute error;
- maximum ratio of observed error to the declared elementwise tolerance;
- PASS/FAIL and a failure reason.

Shape mismatch, non-numeric outputs, and NaN/Inf fail closed.

## Evidence boundary

Replay is an **extraction correctness** witness. It does not prove that the
representation is compressed, that a task-quality target is met, or that a
codec is within the 64x bit budget.

A real-model pipeline should use replay before structural diagnostics and codec
optimization so an extraction bug cannot masquerade as interesting geometry.

## Recommended order

```text
checkpoint + adapter plan
  -> D57 manifest
  -> extracted primitive replay
  -> structural diagnostics
  -> functional sensitivity
  -> codec candidate generation
  -> exact serialized-bit accounting
  -> task-quality evaluation
```
