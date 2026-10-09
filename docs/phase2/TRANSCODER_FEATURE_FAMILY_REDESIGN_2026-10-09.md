# Transcoder feature family pause — 2026-10-09

## Decision

Pause MA-535 through MA-538 while they use the same published SmolLM2-135M layer-8 skip-transcoder atom premise. Resume only after the feature representation or normalization is materially redesigned and that change is preregistered. Keep all four registry rows UNTESTED; this is a queue deferral, not an experimental failure for those IDs.

## Evidence

- **MA-533:** On 2,048 held-out Wikitext-2 train activation vectors, full top-128 transcoder raw output relative MSE was 0.806 and cosine 0.738, missing the <=0.10/>=0.95 fidelity gates. Fit-only norm prediction lowered MSE to 0.149 but still missed the gate. Rank-128 affine control scored 0.817 at 592,381 incremental bytes, while full transcoder state was 341,363,524 bytes. Fresh remained sealed.
- **MA-534:** Using the same checkpoint/layer but a 32-atom shared bank and four fit-derived roles, role Givens Mirror MSE was 0.956. Plain role sparse gates were 0.960, only a 0.4% improvement; a simple diagonal gain control was better at 0.940. The selected bank saved compute and a small amount of standalone storage, with substantial quality loss. Fresh remained sealed.

## Shared structural explanation

**Fact:** Increasing the active set to the released top-128 representation still leaves a large native-MLP output residual. Restricting the physical basis to 32 atoms makes the model smaller and faster, but the approximation degrades further. Small role-specific Givens addresses do not recover the missing directions, and a simpler diagonal gain control performs better.

**Interpretation:** The tested pretrained feature dictionary is not sufficiently aligned or complete for these dense layer-8 MLP outputs. The same basis deficit is the common cause behind both scoped failures, rather than one isolated angle setting.

**Hypothesis for a redesign:** A new experiment could test a layer-specific trained dictionary, explicit output-scale normalization, or private residual atoms. Such work must compare against equal-byte low-rank/diagonal controls and use non-overlapping fit/eval data; it must not reuse the current frozen validation split without a new protocol.

This pause applies only to the MA-533/534 checkpoint/layer feature-basis line. It does not establish that other transcoders, layers, datasets, or sparse circuits fail. Continue with the distinct next P0 candidate MA-539.
