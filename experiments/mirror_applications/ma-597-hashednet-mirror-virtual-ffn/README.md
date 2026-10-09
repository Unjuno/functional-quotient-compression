# MA-597 — HashedNet Mirror virtual FFN

Status: **FAIL**. PA120. Dedicated branch: `research/ma-597-hashednet-mirror-virtual-ffn-20261009`.

## H — Hypothesis

Thirty-two learned input Givens angles should reduce harmful collisions in a 2,048-bucket HashedNet FFN, giving ≥1.0 percentage point test-accuracy gain over native hashing at ≤1.10× native bytes and beating byte-near diagonal and rank-one controls.

## T — Execution

Train a 64→128 ReLU→10 MLP on sklearn digits 1.8.0. Development seeds 59701/59702 use stratified 75/25 splits. Dense, 2,048/4,096 bucket HashedNet, Givens Mirror, diagonal gate, and rank-one residual each receive 800 AdamW updates. Inference loads serialized FP16 NPZ payloads; all learned tensors, hash seed, schema and shapes are charged. Fresh seeds 59711–59713 were sealed.

## D — Decision

**FAIL.** Mirror accuracy was 97.56%/98.22% at 9,753 B; native 2,048-bucket accuracy was 97.56%/98.67% at 9,441 B. The Mirror misses the +1-point gate on both splits. A byte-near diagonal gate tied/beat it, and rank-one residual tied/beat it within 1.10× bytes. The 4,096-bucket native control was 98.44%/98.67% at 13,537 B and reduced collision rate. Mirror training was slower than native hash; short CPU inference timings are noisy.

## C — Counter-hypothesis

Native bucket expansion and ordinary diagonal or rank-one conditioning explain the observed performance. The rotation is an extra learned coordinate without demonstrated Pareto gain.

## U — Limits

One small image dataset, one FFN matrix, two development splits, and fixed-update training do not establish language-model capacity or near-convergence behavior. Fresh splits remained unopened after the gate miss.

See `RESULTS.md` and `verification_report.json` for facts, interpretation, hypothesis, actual payload sizes, collision diagnostics, and exact replay provenance.
