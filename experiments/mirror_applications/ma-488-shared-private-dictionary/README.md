# MA-488 — Shared/private function dictionary

Status: **FAIL** for Mirror-specific attribution (development screen; fresh sealed)  
Branch: `research/ma-488-shared-private-function-dictionary-20261008`  
Base commit: `3a40be6c4300b505d75fe0cf42a52b069ef58ff9`  
Prior art: PA94 (shared/private dictionary learning)

## H — Falsifiable hypothesis

At a nonzero heterogeneity level, shared atoms plus a thresholded private residual should retain heldout function quality at <=0.80x the full-int8 payload, while a sweep identifies where private residual state overtakes the int8 baseline.

## T — What ran

Two frozen worlds (48801, 48802), each at heterogeneity p=0, .125, .25, .5. Four shared 16x16 atoms and 192 linear functions; 128 train, 64 heldout and 64 common query vectors per function. Private functions received an orthogonal full-matrix residual of norm 0.25–0.35. A frozen L2>0.10 gate stored all residuals. Compared shared-only, shared/private, byte-identical native shared/private, full int8 matrices and independent float matrices. All private state and indices were included in actual NPZ payloads. Replay and tests passed; fresh seeds 48811–48813 remain sealed.

## D — Decision

**FAIL** for Mirror-specific attribution. The shared/private bank met the storage/quality gate at p=.125: 33,238 B, RMSE 0 versus 50,074 B full int8, a 33.6% saving. But its payload exceeded full int8 at p=.25 (57,837 B vs 50,073 B), and its payload/hash/output exactly matched the native shared/private control. The result maps an ordinary shared/private storage boundary; it does not establish Mirror-specific value.

## Facts

- p=0: shared-only is lossless at 8,098–8,099 B; no private vectors are allocated.
- p=.125: 24/192 full residuals are stored; shared/private is lossless at 33,238 B. Shared-only private-heldout RMSE is 0.07796/0.07833 (overall 0.03082/0.03392).
- p=.25: 48/192 residuals are stored; shared/private is lossless at 57,837 B, above full int8 50,073 B. Shared-only private-heldout RMSE is 0.07760/0.07497.
- p=.5: 96/192 residuals are stored; shared/private is 107,036 B versus about 50,072 B full int8. Shared-only private-heldout RMSE is 0.07430/0.07460.
- Shared/private payload uses 8,636 B at p=0, then increases with every private vector. Native shared/private bytes/hash and outputs match exactly for every rate. Full replay maximum difference is zero.

## Interpretations

With this explicit full-vector fallback, private residual state is byte-efficient up to the tested 12.5% rate against global int8 matrices. The crossover occurs between 12.5% and 25% heterogeneity. Shared-only codes fail on off-basis functions; private values restore them exactly. This quantifies a private-parameter boundary for the toy family, using a standard native shared/private representation.

## C — Strongest counter-hypothesis

Private residuals are deliberately full precision and a global int8 baseline is strong; per-function low-rank or entropy-coded private controls may move the crossover. The known shared dictionary is also a favorable prior, not evidence that a learned natural dictionary transfers.

## U — Still unknown

Natural task distributions, learned shared bases, low-rank private residuals, runtime on accelerators and online allocation remain untested. No independent capacity follows from nominal logical function counts.
