# MA-476 — GRACE key/radius memory with Mirror-compressed values

Status: **PROMISING for aligned value storage; small Mirror-specific byte margin**
Evidence lane: codebook value compression / retention / locality / bytes / latency
Branch: `research/ma-476-grace-value-mirror-20261009`
Protocol frozen: `5b178cdd`; fresh worlds 47610–47612 × seeds 0–2.

## H — Hypothesis

Keeping GRACE keys and per-key deferral radii intact, a shared value basis and one continuous angle per value can preserve retrieval and counterfactual output while using fewer bytes than explicit values, generic Cartesian shared-plane coefficients and fixed-size codebooks.

## T — Test

An analytic 16D external memory held 64 unit keys and 64 fixed-norm value vectors on a known shared 2D plane. Every method stored the same keys, per-key radii sampled from [0.87, 0.93], and IDs. Queries were exact keys, four Gaussian-noise paraphrases per key (sigma 0.03), and 256 unrelated random keys. Controls were no edit, full GRACE key/radius/value storage, Cartesian coefficients on the shared value plane, a continuous Mirror angle, and uniform discrete value codebooks K=8/16/32/64 with stored codebooks and per-entry indices. Search and selected-value decoding were timed together; payloads included all retrieval and value state.

## D — PROMISING for aligned value compression

**Facts:** Across nine fresh world/seed pairs, every value representation had 100% exact and paraphrase hit recall and zero unrelated false triggers. At N=64, GRACE used 10,973B, generic shared coefficients 7,705B, Mirror 7,449B, and VQ K=8/16/32/64 used 7,641/8,153/9,177/11,225B. Mirror value NRMSE was below 1.3e-7. The VQ value NRMSE was 0.249, 0.118, 0.060 and 0.031, respectively. Mirror used 67.9% of explicit GRACE bytes and was 3.3% smaller than the generic coefficients. At N=1, Mirror was 2,777B vs explicit values at 2,525B. At N=64 median lookup-plus-selected-decode was 0.71 microseconds/query for Mirror, 0.55 for generic coefficients and 0.49 for explicit values on this CPU probe.

**Interpretation:** A known continuous value orbit gives exact outputs at lower bytes than the tested discrete codebooks, which trade more codebook bytes for quantization error. The shared basis accounts for most savings versus GRACE. The incremental Mirror gain over the generic shared-plane code is only 3.3%, and it adds measured decode time; this is a narrow storage frontier, not a broad GRACE improvement.

## C — Strongest counter-hypothesis

The continuous angle is ordinary polar coding on a hand-specified fixed-norm circle. Natural GRACE values may not share a low-dimensional orbit, and a learned/adaptive codebook could beat this fixed uniform quantizer. The scope radius and key state dominate some smaller-N payloads.

## U — Unknown

Learned GRACE key/value adaptation, natural edit values, distribution-shift false triggers, sequential lifelong retention, learned codebook training, and deployed serving cost are untested. The fixed thresholded cosine lookup is not GRACE's learned locality mechanism.

## Decision

**FACT:** At N=64, angle codes preserve synthetic retrieval/value quality with fewer bytes than explicit GRACE, generic Cartesian coefficients and the tested VQ frontiers. N=1 has no storage advantage; Mirror lookup/decode is slower in this CPU microbenchmark.
**INTERPRETATION:** The result is an aligned fixed-norm value-memory Pareto point with a small incremental margin over a simple shared basis.
**HYPOTHESIS:** Natural edit values may have structure that allows a similar value-side compression while GRACE key locality remains unchanged.
**BOUNDARY:** Synthetic hand-aligned value memory; no natural GRACE or continual-learning evidence.
