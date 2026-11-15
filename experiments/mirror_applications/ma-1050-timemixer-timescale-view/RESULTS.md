# MA-1050 — TimeMixer seasonal-timescale Mirror screen

Status: FAIL (scoped fixed-budget screen)

## H
A low-description scale×regime Mirror coordinate bank over a shared multiscale mixer would reduce audit error by at least 2% against hard sharing and beat a simple additive control.

## T
Pinned Time-Series-Library `4e938a1767106324dd753b2a44832bf870a0252e` official source; ETT dataset commit `a52c4925778c07c1ef1a2cf6fd01594919717d9e` / `62c9f979e2dc5b05d9cc5b38891e60e6baf77417`. Targets were OT in ETTh1/ETTm1, chronological 60/20/20, context 96/horizon24, 48 train origins/domain, 12 dev and 12 audit origins/domain. Seed 105001; 20 epochs, 40 AdamW updates per method, same initialization and example budget. The local model is a CPU-scale multiscale mixer adapter, not an official full TimeMixer reproduction. ETT used in earlier program experiments, so audit is temporal holdout only, not a fresh dataset.

## D — FAIL (screening only)
```
method       dev MSE     audit MSE    payload bytes   train sec
hard-share   0.038897    0.043438    205872       0.572
Mirror       0.039147    0.044012    206124       0.518
additive     0.040178    0.043244    207946       0.586
```

Mirror audit MSE is 1.32% worse than hard sharing and its payload is 252 B larger. Additive control is 0.45% better than hard sharing and has a larger payload. Neither meets the 2% gate; no Mirror-specific benefit established. This fixed-budget result does not establish capacity.

## C — strongest counter-hypothesis
The small CPU-scale adapter and single seed may confound architectural behavior; also the additive method is not exactly parameter matched to the Mirror function family. This is an implementation/mechanism screen, not a definitive test of full TimeMixer.

## U — unresolved
Independent per-domain upper control, additional seeds, full official TimeMixer training pipeline, broader seasonal regimes, true byte-normalized Pareto sweeps, and accelerator runtime remain untested.

FACT: this run produced the metrics/bytes above. INTERPRETATION: Mirror did not improve the fixed-budget frontier over hard sharing or additive control. HYPOTHESIS: richer learned multiscale coefficients may need more data/updates or private residuals.
