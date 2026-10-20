# MA-1173 — Lookahead Structural Interventions in a Mirror Virtual Supergraph LM

## Central hypothesis
**Mirror insertion:** generate 18 ordered virtual 12→12→12 edge-MLPs from one trainable 288-scalar physical core plus learned address and gate coefficients, then alter the binary edge topology *during learning* to influence the future function learned by the shared core, rather than pruning based solely on immediate weakness.

A small synthetic **causal token-prediction** task tests the first mechanism; this is **not a natural-language-model demonstration** and does not prove arbitrary MLP expressivity. The eventual full research goal includes node count, Mirror directions, addresses, recurrent passes and output methods. These are held constant in this first controlled experiment to isolate *prospective edge interventions*.

## Controls
- Mirror full graph unchanged, random two-edge cut, weakest-gate two-edge cut, future-loss lookahead cut, future-loss guarded cut.
- Exact shared-address **direct tanh** and **affine** future-loss lookahead structural controls.
- Native 12→16→12 tied recurrent FFN with near-equal trainable coefficients.
- Mirror no-cut with extra 120 real training updates to roughly match the lookahead candidate-probe optimizer step budget.

## Study scope and current stage
The nine-model × two-world × three-seed comparison will hold future fresh data until all optimization and cutoff choices are frozen. Every decision is made on development examples; total computation of speculative lookahead is counted. Each cut event removes two of 18 graph edges (not independent physical weights); only after repeated cut events do active graph edges reduce to 12. Actual torch serialized inference bytes are reported rather than raw scalar-count estimates.

**Mirror-specificity is an open question.** A quality improvement against weaker graphs does not certify Mirror advantage if non-Mirror address sharing reproduces it.

`PROTOCOL.json` is frozen before training. `source/model.py`, `source/data.py`, and `tests/test_model.py` include pretraining L0 checks. Training audit will be written to `RESULTS_CORE.csv`, `REPORT_JA.md`, `VERIFICATION.json` and `checkpoints/`.
