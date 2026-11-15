# MA-1046 — Chronos distribution View

**Decision: FAIL** for the registered quality gate and Mirror-specific attribution. The two-coordinate View underperforms zero-shot Chronos and native temperature on normalized CRPS on both fresh datasets, and the normalized-Hermite control reproduces its result within floating-point noise.

The complete frozen protocol, amendments, model/data provenance, raw forecast archive, and actual payload accounting are preserved here. Start with [PROTOCOL.json](PROTOCOL.json), [RESULTS.md](RESULTS.md), and [VERIFICATION.json](VERIFICATION.json).

The development/fresh split uses ETTh1 and ETTm1 only for fitting/selection, then Weather and Australian Electricity for the fresh sampling audit (seeds 104604–104606). The final report reruns from saved forecast samples without repeating model inference. Earlier invalidated Weather forecasts and reducer fixes are disclosed in the amendment history and `runs/`.
