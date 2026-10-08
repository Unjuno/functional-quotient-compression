# MA-715 status

- Status: **FAIL** (development quality/storage gate missed)
- Branch: `research/ma-715-regmean-mirror-20261008`
- Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
- Draw 11 replay: passed; pool 557, seed/index in `source/random_draw.json`
- Protocol: frozen and committed before dataset values or scores were inspected
- Development: complete; 2 worlds × 16 requests, 14 method variants, ranks 1–4
- Audit/test partition: **not accessed**; both worlds failed quality/storage before the Mirror-specific gate
- Actual payloads: measured and replay checked; max float32 reconstruction delta <= 5e-8
- Tests: 5 passed
- Result / verification: recorded in `README.md`, `RESULTS_CORE.csv`, and `VERIFICATION.json`

## H / T / D / C / U

- **H:** RegMean statistics could produce compact Mirror coefficients that preserve mixture quality and beat byte-near controls.
- **T:** Two frozen sklearn digits worlds, four rotated ridge sources, rank-1..4 direct-stat solve plus full RegMean, task arithmetic, source projection, and train-output PCA controls; 16 development mixtures per world.
- **D:** **FAIL.** No rank met quality/storage. Rank 4 saved 65.0% inference bytes but lost 19.3–19.6 percentage points accuracy versus full RegMean. At equal bytes, output PCA was substantially better. Audit unopened.
- **C:** The RegMean mixture functions are poorly aligned with the source-delta basis; output-PCA fits the required output family more directly.
- **U:** Fresh/audit generalization, nonlinear/large models, larger or learned bases, private residual tradeoffs, and optimized runtime remain untested.

## Blockers

None. The result is a scientific FAIL, not a hardware or dependency blocker.
