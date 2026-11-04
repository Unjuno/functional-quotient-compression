# MA-673 status

- Status: FAIL (registered byte gate and Mirror-specific claim)
- Branch: `research/ma-673-task-layer-lora-code-20261104`
- Base commit: `c7c4d204`; corrected development harness amendment recorded before metrics
- Fresh/audit: sealed

## H — Hypothesis

A task factor and layer factor can represent task-specific rank-4 adapter coordinates across layers, preserving held-out task-layer outputs while using at most 60% of independent coefficient-table bytes. Mirror must beat native outer-product factorization.

## T — Treatment and controls

Synthetic D=24, 12 tasks, 6 layers, rank4, 96 inputs per pair; two development seeds. Measured held-out task/layer pair NRMSE and actual compressed NPZ sizes for independent coefficient tables, task×layer product codes, and native outer-product factors. A held-out indexing error was corrected before any metrics were generated and disclosed in amended protocol/freeze metadata.

## D — Decision

FAIL. Product codes reconstruct the generated separable functions exactly (NRMSE 0) on the selected held-out pairs, but actual payloads are only ~52.2KB versus ~52.8KB independent, far above the 60% byte gate. The native outer-product control is 6B smaller than the Mirror field encoding in both seeds and has identical factors/function. No fresh runs were opened.

## C — Strongest counter-hypothesis

The tested factorization is ordinary task×layer outer-product coefficient sharing. The serialized rank-4 factors save too little versus the independent table at this small tensor shape and offer no Mirror-specific behavior.

## U — Unconfirmed

No learned coefficients, larger banks, natural task-layer generalization, GPU performance or latency claim is established. The synthetic generator supplies exactly separable factors, so this does not establish general compositional capacity.

## Evidence labels

- Fact: exact output errors and NPZ sizes are in `RESULTS_CORE.csv` and `source/development.json`.
- Interpretation: this tensor shape does not amortize the factor representation; native outer-product coding is marginally more compact.
- Hypothesis: task-layer factorization may only amortize at larger task counts/ranks, but native factor controls remain mandatory.
