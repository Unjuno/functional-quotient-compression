# MA-715 Results — RegMean statistics to Mirror merge coefficients

## H — Hypothesis

For mixtures of four rotation-specific digit classifiers, direct Mirror coefficient solving from RegMean sufficient statistics would preserve development accuracy/NLL within 1 percentage point / 0.05 of full RegMean while reducing actual 16-function inference payload to at most 70%, and would beat byte-near simple controls.

## T — Training and evaluation

Frozen before fitting: sklearn `load_digits` v1.8.0; two stratified 60/20/20 worlds (seeds 71501, 71502); multiclass ridge sources from four fixed rotations; 8 training mixtures for output-basis construction; 16 development mixtures per world; rank sweep 1–4. Controls were full RegMean, task arithmetic, source-delta projection, and output-merge PCA. The 32 audit mixtures per world were not generated and audit indices were not opened because development gates failed. Five preflight/unit tests pass. CPU-only single-thread run; no optimizer updates.

## D — FAIL

### Facts

- No common rank passed the preregistered quality/storage gate in both development worlds; audit remained locked.
- Full RegMean achieved accuracy/NLL 0.7758/2.0276 and 0.7664/2.0440 in the two worlds; its actual serialized 16-model inference payload was 42,972 bytes.
- At rank 4, direct statistics-to-Mirror achieved 0.5797/2.0999 and 0.5731/2.1150, with 15,060-byte payloads (35.1% of full payload; 65.0% fewer bytes). Relative accuracy gaps were -19.61 and -19.33 percentage points; NLL exceeded the allowed margin by 0.0223 and 0.0210.
- At the same 15,060 bytes, the simple output-PCA basis achieved 0.7518/2.0409 and 0.7422/2.0532. Thus it was markedly more accurate and had lower NLL than direct-stat Mirror in both worlds.
- Rank-4 direct-stat merge MAC proxy was 189,604 versus 336,375 for full RegMean. Measured per-request CPU merge wall time was 0.319 ms / 0.387 ms for direct-stat Mirror and 0.159 ms / 0.163 ms for full RegMean; the short Python timing is a mechanism screen, not a serving benchmark.
- Task arithmetic used a smaller 12,312-byte inference library but achieved only 0.5705/2.1046 and 0.5650/2.1194. It does not establish a useful quality/storage frontier point.
- Payload replay maximum difference was at most 5e-8 in float32. Task arithmetic showed a one-ULP-scale reduction-order difference; the tolerance and observed difference are retained in source metadata.

### Interpretation

The direct statistics solve did compress the 16 outputs, but the available rank-4 source-delta subspace did not retain their useful mixture functions. The byte-matched output-PCA control captured the merge outputs much better. Lower symbolic MAC proxy did not translate to lower measured CPU wall time in this unoptimized implementation. No capacity or general-model claim follows from this screen.

### C — Strongest counter-hypothesis

RegMean output variation lies outside the span of the four source-model deltas. The result is therefore a basis mismatch, not evidence that solving coordinates from sufficient statistics is inherently unhelpful; however, output-PCA already supplies the cheaper and stronger byte-matched control under this protocol.

### U — Unconfirmed

Audit/fresh performance, other datasets, nonlinear models, larger or learned bases, convergence behavior, optimized kernels, and whether an additional paid private residual can recover quality are untested.

## Artifacts

`RESULTS_CORE.csv` holds all 448 method/request rows (2 worlds × 16 mixtures × 14 methods); `source/development_summary.json` and `source/development_raw.json` preserve the per-world outcomes; split manifests retain audit indices as locked metadata. Actual serialized payloads and merge packages are under `source/artifacts/`.
