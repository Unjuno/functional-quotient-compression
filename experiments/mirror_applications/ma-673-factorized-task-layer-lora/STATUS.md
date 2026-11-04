# MA-673 status

- Status: FAIL (byte gate and Mirror-specific claim)
- Branch: `research/ma-673-task-layer-lora-code-20261104`
- Base: `c7c4d204`; corrected protocol amendment recorded before rerun
- Development: seeds 67301/67302; fresh sealed

## H — Hypothesis

Task and layer factors over a shared rank-4 LoRA atom basis preserve held-out task-layer outputs at <=60% of independent coefficient-table bytes. Mirror must beat native outer-product factorization.

## T — Treatment and controls

Synthetic D=24, T=12, L=6, R=4 with true rank-1 LoRA atoms; measured held-out pair NRMSE, compressed NPZ sizes, MAC proxy and CPU wall time. Compared independent task-layer coefficients, task×layer product coordinates and native outer-product factors. The initial dense-atom implementation is preserved at `source/attempts/initial_dense_atom_basis/`; its protocol amendment replaced D×D atoms and fixed-zero wall-time reporting with rank-1 factors and timed execution before rerun.

## D — Decision

FAIL. All methods reconstruct the separable held-out functions exactly. Corrected product payloads were 5,474–5,479B versus 6,115–6,117B independent, or about 90%, missing the preregistered 60% gate. Native outer-product files were 6B smaller and function-identical. Active MAC proxies were effectively equal (product adds a small coefficient multiply); observed CPU times were noisy at sub-millisecond scale. Fresh remained sealed.

## C — Strongest counter-hypothesis

The task×layer product is ordinary outer-product coefficient sharing. It saves only a small fraction at this bank shape and has no Mirror-specific behavior.

## U — Unconfirmed

Learned task/layer factors, larger banks, natural adapter generalization, and meaningful hardware latency are untested. This synthetic separable generator cannot establish unseen task capacity.

## Evidence labels

- Fact: NPZ sizes, replayed held-out errors, MAC proxies, and measured wall times are in `RESULTS_CORE.csv` and `source/development.json`.
- Interpretation: the corrected low-rank LoRA factor version still misses the storage gate and aliases native outer products.
- Hypothesis: larger T/L ratios may amortize factors, but any such test must compare ordinary native factorization.
