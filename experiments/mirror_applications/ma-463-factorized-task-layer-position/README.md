# MA-463 — Factorized task × layer × position Mirror codes

Status: **FAIL**
Evidence lane: FACTOR COMPOSITION / HELD OUT COMBINATIONS / QUALITY / BYTES
A1 freeze: `f5f9485a`; valid fresh worlds 46320–46322. Initial 46310–46312 results remain in `artifacts/exploratory_first_attempt_summary.json` and are excluded because factor initialization was not explicitly keyed to world/seed.

## H — Hypothesis

A product of task, layer, and adapter-position Mirror scalar codes can recover held-out factor combinations with similar quality to HyperFormer while using at most 60% of its actual payload.

## T — Test

Synthetic 2×2 adapters on 8 task × 4 layer × 3 position factors (96 combinations). Teacher targets were `A₀ + (u_task v_layer w_position) B`. The split withheld 24 structured combinations while all factor values appeared in training. Controls were a HyperFormer MLP over concatenated one-hot IDs, ordinary rank-one CP product factors, additive linear hypernetwork, and independent matrices. Each combination had 64 support and 256 query examples. Development selected 500 updates. A1 fixed initialization to world/seed and used three replacement fresh worlds × three seeds. Serialized factor tables, indices, decoder weights, and metadata were charged.

## D — FAIL

**Fact:** At N=96, mean held-out NRMSE was 0.3064 Mirror, exactly 0.3064 ordinary CP, 0.1559 HyperFormer, 0.3775 additive, and 1.75e-7 independent upper control. Actual bytes/combo were 33.55 Mirror/CP, 55.59 HyperFormer, 25.72 additive, and 39.05 independent. Mirror used 60.4% of HyperFormer bytes, just missing the 60% gate, and missed the quality gate. In one fresh world the mean Mirror error was 0.919 vs HyperFormer 0.149 due to one factor-fit seed failing badly. Mirror and CP have identical output metrics and serialized payload hashes for every seed and N.

**Interpretation:** The factor product is a useful compact representation for this teacher when optimization converges, but it is exactly ordinary rank-one CP parameterization. Training is fragile across seeds, and the simpler additive control is smaller with slightly better mean error. No Mirror-specific gain is established.

## C — Strongest counter-hypothesis

The task family is exactly a rank-one tensor product, so CP factorization is the natural generic explanation; observed failures are optimization/identifiability issues rather than Mirror geometry.

## U — Unknown

Natural task/layer/position adapter variation, higher-rank interactions, Transformer fine-tuning, and more robust CP initialization remain untested.
