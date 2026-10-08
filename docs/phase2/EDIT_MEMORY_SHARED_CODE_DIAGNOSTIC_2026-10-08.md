# Edit-memory shared-code diagnostic — 2026-10-08

## Scope

This diagnostic covers three frozen synthetic external/weight-edit storage candidates: MA-475 (SERAC-style values), MA-476 (GRACE-style value codebook), and MA-478 (View-first with private fallback). It does not cover factual model editing or logical function codebooks.

## Results (facts)

| MA | Evidence | Strongest ordinary control | Outcome |
|---|---|---|---|
| 475 | 256 rank-eight values: 44,483 B, RMSE <=1.1e-7 vs explicit 99,513 B | Native PCA has the same bytes/hash/output; int8 is 51,648 B at RMSE about 0.001 | FAIL attribution |
| 476 | VQ16/64/128: 37,298–40,883 B, RMSE 0.111–0.142 | Latent int8 is 38,623 B at RMSE <=0.01191; continuous PCA is a native alias | FAIL quality/attribution |
| 478 | 16/256 heldout values needed private fallback; adaptive bank 49,350 B and RMSE 0 | Native PCA plus the same residual gate exactly aliases; frozen ratio to int8 misses by 0.6 percentage points | FAIL attribution |

All three used fixed key routing, paid key state, exact serialized byte counts and replay verification. These were synthetic, deliberately structured value families. Fresh seeds remained sealed after the frozen gates failed.

## Interpretation

The measured storage gains are real for their stated toy tasks, but the selected Mirror coordinates add no distinct function family: native PCA, ordinary VQ/int8 quantization, or low-rank-plus-private residual reproduces the storage mechanism. The fallback experiment does show that off-basis values can require private state: in its heldout split, the 16 off-basis values were exactly the 16 values selected by the predeclared residual threshold.

This is evidence to compare future external-memory work directly with standard representation compression and to report private residual fractions. It is not evidence against logical function addresses, learned routing, natural factual memories, or sequential model edits. MA-481/482 change the target from external edit values to a bank of logical functions and are treated as a different candidate family.

## Hypotheses for later work

- Value compression should beat native PCA/int8 at matched quality, or identify a distinct operational advantage such as reversibility, edit composition, or selective privacy.
- A private residual can recover off-basis functions, but its actual bytes and write/read cost must be charged.
- Natural edit collections may have different shared/private spectra than the exactly aligned synthetic banks used here.
