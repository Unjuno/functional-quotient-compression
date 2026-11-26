# MA-1063 — DHE with structured role Mirror views

## H — Hypothesis

On MovieLens 1M binary preference classification, adding sixteen learned Givens angles per user/item role to a shared table-free DHE decoder would improve fresh AUC by at least 0.005 over the strongest matched diagonal, no-view, or FiLM DHE control on both training seeds, remain within 0.01 AUC of full tables, use at most 25% of their complete serialized payload, and retain at least 80% of no-view DHE throughput.

## T — Trial

Used the official GroupLens MovieLens 1M archive (SHA256 `a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20`, 5,917,549 bytes; access terms and citation in `DATASET_TERMS.md`). The deterministic SHA256(user ID, item ID) modulo 10 split produced 800,787 train, 99,704 development, and 99,718 audit ratings with no user/item-pair overlap. Rating >=4 is positive. The four-hidden-layer width-64 DHE decoder used k=128 dense hash features, a documented scaled CPU screen rather than a reproduction of the paper's k=1024 setup.

Six controls: independent full-cardinality tables; shared DHE; shared DHE plus matched 16-value paired diagonal; shared DHE plus 32-value FiLM; shared DHE plus 16-angle Givens Mirror per role; and independent user/item DHE decoders. Each method selected one of epochs 1–3 by mean development logloss over seeds 106300/106301; all selected epoch 3. Fresh training seeds were 106310/106311. Audit access was frozen in `FRESH_FREEZE.json` before scoring. No settings were changed after audit.

## Fact — frozen fresh audit

Means over two training seeds on the same held-out audit split. Payload bytes are actual serialized model file plus inference metadata. QPS is measured CPU batch throughput (batch 1,024, 40 repeats); it is not a GPU or production serving benchmark.

| Method | Logloss ↓ | AUC ↑ | Payload bytes | QPS | p95 ms | Train seconds | Rare-item AUC (<=5 train ratings) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full tables | 0.545652 | 0.789620 | 1,282,540 | 17,035,162 | 0.204 | 2.03 | 0.656606 |
| Shared DHE | 0.555356 | 0.777046 | 106,783 | 218,410 | 5.984 | 35.56 | 0.611515 |
| DHE + paired diagonal | 0.555343 | **0.777068** | 107,158 | 211,547 | 6.033 | 33.17 | 0.608848 |
| DHE + FiLM | 0.555786 | 0.776608 | 107,542 | 230,030 | 5.530 | 29.51 | 0.605212 |
| DHE + Mirror | 0.555465 | 0.776929 | 107,238 | 203,838 | 6.418 | 32.57 | 0.606424 |
| Independent DHE | 0.547320 | 0.783867 | 211,978 | 208,079 | 6.384 | 34.90 | **0.649455** |

Mirror uses 8.36% of full-table payload (91.64% fewer bytes), satisfying the 25% storage limit. It trails full tables by 0.012691 AUC, missing the 0.01 tolerance. Its mean AUC is 0.000139 below the matched diagonal; its logloss is 0.000122 worse. Both fresh seeds miss the required +0.005 Mirror advantage. Mirror throughput is 93.3% of no-view DHE on average and passes the 80% throughput gate, although it is 6.7% slower than shared DHE in this CPU measurement. Independent DHE substantially recovers quality over the shared decoder while using 211,978 B, showing that decoder sharing has a quality cost.

The audit's item-frequency <=5 slice has 130 rows; Mirror AUC is 0.606424 versus 0.608848 for matched diagonal and 0.649455 for independent DHE. It does not rescue the claim. All controls used the same updates per selected epoch (588); the short three-epoch screen is not near-convergence evidence.

## D — Decision

**FAIL** for the registered quality/storage/compute gate. The table-free DHE baseline gives a strong storage reduction, but Mirror fails both fresh-seed quality-margin requirements and the full-table AUC tolerance. The byte-matched diagonal is slightly better. No Mirror-specific benefit or capacity increase is established.

## C — Strongest counter-hypothesis

The dominant gain is table-free deterministic hashing and a learned decoder. The small rotation coordinate adds almost no useful freedom after short training; the measured differences from the paired diagonal are within ordinary seed variation. More decoder capacity or longer training may improve all shared-DHE variants, but that would not overturn this frozen Mirror-versus-diagonal comparison.

## U — Unconfirmed

This is one MovieLens dataset/split, two model seeds, a compact k=128 implementation, and CPU-only inference timing. It does not establish a production DHE result, GPU bandwidth behavior, cold-start generalization, near-convergence capacity, or statistical generalization across datasets. The 130-row rare-item slice is small. Results, model SHA256s, serialized replay, and settings are detailed in `runs/audit/results.json`, `runs/audit/verification.json`, and `FRESH_FREEZE.json`; raw ratings and model checkpoints are not committed.

## Interpretation and next test

**Fact:** Shared DHE compresses the full tables by 91.7% but has lower AUC. The Mirror variant barely changes its behavior and does not beat its same-size diagonal control.

**Interpretation:** DHE's deterministic hash decoder accounts for the observed storage frontier. In this screen, adding role-specific orthogonal coordinates does not produce useful logical multiplicity beyond a simple diagonal control.

**Hypothesis:** Private exceptions for rare IDs may recover the observed long-tail gap more efficiently than global role rotations. That is the registered next candidate MA-1064 and will require a fresh protocol and its own branch.
