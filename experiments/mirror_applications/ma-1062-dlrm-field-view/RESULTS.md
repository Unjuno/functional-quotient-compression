# MA-1062 — DLRM feature-field Mirror views

## H — Hypothesis

On MovieLens 100K explicit rating classification, a shared H=512 embedding table with eight field-wise 2D Givens angles should beat the strongest shared-hash/diagonal control by at least 0.002 fresh logloss on both training seeds, stay within 0.005 AUC of full-cardinality DLRM, reduce complete payload by 25% or more, and keep lookup p95 within 1.25× ordinary H=512 hashing.

## T — Trial

Pinned GroupLens MovieLens 100K archive (SHA256 `50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229`; 4,924,029 bytes). Official `ua.base` supplied 90,570 training-pool ratings, split by SHA256(user,movie) into 81,490 train and 9,080 dev with no pair overlap. Each method selected epoch 3/5/8 independently by mean dev logloss across seeds 106200/106201. Audit used official `ua.test` (9,430 ratings) and seeds 106210/106211. The audit split was frozen before reading; first audit invocation failed before model construction, update or scoring, and AM-02 documents the retry. No audit rows/labels/IDs were persisted; only logits, aggregate metrics and model payload artifacts were retained locally.

Controls: independent full-cardinality tables; shared hash H=128 and H=512; four independent 128-row field tables; paired-diagonal controls at H=128/512; per-coordinate H=512 diagonal; Mirror Givens views at H=128/512. All use the same compact DLRM MLP dimensions. Field views are learned jointly with the table. MovieLens access terms and citation are in `DATASET_TERMS.md`; raw archive was not committed.

## Fact — audit results

Lower logloss is better; higher AUC is better. Values are means across the two model-training seeds on the same official audit split. Actual payload bytes include the full learned network/tables, coordinates, timestamp normalization and metadata.

| Method | Logloss | AUC | Complete payload bytes | Lookup p95 ms (seed range) |
|---|---:|---:|---:|---:|
| full | 0.571250 | 0.765202 | 187,591 | 0.107–0.167 |
| shared_hash_128 | 0.645282 | 0.651758 | 25,769 | 0.181–0.211 |
| shared_hash_512 | 0.607905 | 0.715037 | 50,345 | 0.153–0.193 |
| field_hash_128x4 | 0.637511 | 0.666806 | 51,243 | 0.239–0.437 |
| mirror_128 | 0.638896 | 0.662864 | 26,019 | 0.784–1.051 |
| diag8_128 | 0.637934 | 0.664104 | 26,002 | 0.577–0.665 |
| diag8_512 | 0.607454 | 0.715962 | 50,578 | 0.558–0.798 |
| diag16_512 | 0.606059 | 0.717075 | 50,723 | 0.331–0.458 |
| mirror_512 | 0.605938 | 0.716810 | 50,595 | 0.830–1.285 |

H=512 Mirror payload is 73.03% smaller than full DLRM, so the storage threshold passes. Versus the strongest byte-near control on audit, paired seeds do not pass the 0.002 margin: Mirror is -0.000558 logloss better/worse than diagonal16 on seed 106210, and +0.000798 on seed 106211 (positive means Mirror lower/better). Mean advantage over diagonal16 is only 0.000120; mean advantage over plain hash512 is 0.001967. Mirror AUC is 0.716810 versus full DLRM 0.765202, a 0.048392 gap, beyond the 0.005 tolerance.

Lookup p95 for H=512 Mirror is 5.44× and 6.65× plain hash512 across the two seeds, beyond the 1.25× gate. End-to-end p95 is also higher in both seeds. The extra trig/rotation work is not free.

Frequency slices use only training-pool user/movie counts. The audit contains 2,130 rows from users with 6–20 training ratings and 7,300 from users with >20; there are no colder user rows. Movie bins contain 2 rows with zero training ratings, 77 with 1–5, 334 with 6–20, and 9,017 with >20. For the largest movie bin (>20), Mirror logloss is approximately 0.6038 vs 0.6040 diagonal16 and 0.5711 full; improvements over diagonal are far below the gate. The 77-row rare-movie slice has Mirror 0.6299 vs diagonal16 0.6255. No frequency segment rescues the registered claim.

Payloads were evaluated after reloading serialized `.pt` state. Independent replay reproduced all 18 method/seed prediction arrays within 2.4e-7 absolute logit difference and all metrics within 2.2e-11. Actual file bytes and SHA256 are recorded in `runs/audit/verification.json`; test row labels and IDs are absent from saved NPZ artifacts.

## D — Decision

**FAIL** for the registered quality/storage/runtime gate. The 73% full-table payload reduction passes, but Mirror fails both fresh-seed quality margin, AUC tolerance, and lookup latency gates. It essentially ties the stronger per-coordinate diagonal and does not establish Mirror-specific value.

## C — Strongest counter-hypothesis

The hash table capacity/collision pattern and short 8-epoch training budget dominate the quality gap. A stronger table-free or collision-aware representation could change the frontier. That does not explain away the direct Mirror-versus-diagonal result, which is nearly identical at matched scale.

## U — Unconfirmed

One MovieLens split, compact in-house DLRM screen, two training seeds, CPU-only lookup latency, fixed epoch selection. No production DLRM reproduction, GPU bandwidth/QPS, independent dataset, near-convergence, cold-start generalization, or capacity claim. The official test labels were used only after freeze and were not saved.

## Interpretation

**Fact:** H=512 Mirror saves 73.0% complete payload against independent full-cardinality tables, but misses fresh quality gates and is 5.4–6.6× slower in measured p95 lookup than plain shared hashing. Strong diagonal controls match it within 0.001 logloss.

**Interpretation:** A shared physical hash table plus field rotations does not create a useful storage/quality/compute Pareto improvement in this task. Most of the byte reduction comes from ordinary hashing; the Mirror code contributes no practically separable quality advantage over diagonal gains.

**Hypothesis:** Larger bases, collision-aware indexing, or better training could improve all shared-hash variants. Any such follow-up should compare the same controls and would be a new experiment, not a revision to this audit.

## Compute and provenance

CPU float32, one Torch thread. Selected fresh epochs are in `runs/DEVELOPMENT_SELECTION.json`; each method trains separately on all 90,570 ua.base rows. Two-seed optimizer updates range from 267 for full DLRM at epoch 3 to 712 for 8-epoch models. Mirror512 training wall time totals 7.78 s across two seeds versus 5.62 s for diagonal16 and 5.48 s for plain shared hash512. FLOPs were not measured. Actual `.pt` + metadata payload byte totals are reported above; model/logit payload file hashes and exact replay results are in verification.
