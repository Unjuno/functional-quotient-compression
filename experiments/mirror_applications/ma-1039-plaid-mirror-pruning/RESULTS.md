# MA-1039 — PLAID centroid pruning with a query-role task View

**Decision: FAIL** for the registered Mirror-specific retrieval hypothesis. The experiment is scoped to one CPU ColBERTv2/PLAID index and a bounded TREC-COVID candidate collection.

## H — Hypothesis

Two development-selected query-role Givens views of the centroid bank will improve fresh TREC-COVID MRR@10 and final Recall@1000 by at least 0.5 percentage points over native PLAID on one fixed residual index, without duplicating that index; they will also beat the matched native query-side inverse-Givens control by at least 0.5 pp MRR.

## T — Execution

- Official ColBERT source commit: `cc4f3dc91c0b45d2d08c251d9d95178285c65f1c`; official ColBERTv2 checkpoint archive SHA256 `d07df4b2a9fe1848fbed333127a1a5b4f2d96106f420948f0db4450b018cb119`.
- Official BEIR TREC-COVID archive SHA256 `120f42a7864d2214234537733c0d2c6684e42fdfafff2c5eacf98afca6656aa0`.
- The fixed collection contains all 17,537 documents with positive qrels plus 512 deterministic hash-selected background documents (18,049 total). Numeric qid residues 0 and 1 modulo 5 form the audit: 30 development queries and 20 fresh audit queries. This is a bounded, qrels-positive candidate collection, not the full TREC-COVID corpus benchmark.
- One 128-D, 1-bit ColBERT residual index with 512 centroids, 2.371M document token vectors, `ncells=2`, centroid threshold 0.45, and `ndocs=4096` was shared by every arm. FAISS k-means used 20 iterations and seed 123. A deterministic 4,096-passage seed-1039 codebook training sample was used as the preregistered CPU-screen amendment; all 18,049 candidates were encoded into the final index. Index build wall time was 3,355.76 s (55:55).
- Short/long roles were split at 16 raw ColBERT tokens. Development selected angles +0.10 and −0.15 rad from the frozen 9-angle grid. No optimizer updates; 270 development angle-query evaluations.
- Audit arms: native PLAID, centroid-bank Mirror, matched native query-side inverse Givens, and full compressed-index MaxSim scan. Each ran once per audit query; latency is therefore descriptive and **NOT ESTABLISHED**.

## Audit results

| Arm | MRR@10 | nDCG@10 | Final Recall@1000 | Pre-pruning IVF-union recall | Mean pre-pruning candidates | Complete payload bytes |
|---|---:|---:|---:|---:|---:|---:|
| Native PLAID | 0.097917 | 0.029629 | 0.052559 | 0.697716 | 12,430.30 | 491,544,697 |
| Mirror centroid View | 0.096726 | 0.029246 | 0.055653 | 0.716217 | 12,772.95 | 491,544,817 |
| Native query inverse-Givens control | 0.096726 | 0.029246 | 0.055653 | 0.716217 | 12,772.95 | 491,544,817 |
| Full compressed-index scan | 0.097143 | 0.028564 | 0.060615 | n/a | 18,049 | 491,544,697 |

Relative to native PLAID, Mirror changes MRR by **−0.119 pp**, nDCG by **−0.038 pp**, and final Recall@1000 by **+0.309 pp**. The pre-pruning union recall rises **+1.850 pp**, with about 343 more documents presented to native pruning/reranking. This candidate-generation increase does not meet the final-recall gate and does not improve MRR.

Mirror and the ordinary inverse query-side Givens control have identical centroid-cell sets, pre-pruning candidate sets, and top-100 rankings on all 20 audit queries. Maximum centroid-score difference is `3.8743e-7`; their MRR, nDCG, final recall, candidate-union recall, and actual payload are identical. Mirror changes the native PLAID top 10 on 9 of 20 queries, but that variation is fully reproduced by the ordinary query-side parameterization.

## Storage and compute

The native complete payload is 491,544,697 B: ColBERTv2 model/config/tokenizer 439,106,417 B, official PLAID index 52,239,740 B, and PID-to-document mapping 198,540 B. The two role angles plus role metadata serialize to 120 B, giving a Mirror total of 491,544,817 B. Mirror adds **120 B** and saves no bytes against the fixed-index utility frontier.

The full scan is materially slower in this single-run CPU sample (median 1.834 s/query) than pruned PLAID (0.086 s native; 0.084 s Mirror). These one-pass values have no repeated isolated calibration; no stable throughput or latency improvement is claimed. The actual centroid View adapter adds a small, dynamic rotation before the unchanged official PLAID route and rerank.

## D — Decision

**FAIL.** Mirror improves pre-pruning candidate-union recall but misses both the registered final MRR and final Recall@1000 thresholds. The exact ordinary query-side control reproduces the Mirror function and all useful measurements. There is no Mirror-specific retrieval, storage, or established runtime advantage in this screen.

## Fact / Interpretation / Hypothesis

**Fact:** In the 20-query audit, Mirror has MRR@10 0.096726, final Recall@1000 0.055653, pre-pruning union recall 0.716217, and complete payload 491,544,817 B. It matches the native query-side control on audit rankings through top 100, candidate sets, metrics, and bytes. Independent metric replay differs by 0.0; payload arithmetic and candidate mapping replay exactly.

**Interpretation:** Rotating the shared centroid bank at scoring time is an implementation of the same orthogonal score change as rotating the query in the opposite direction. The extra coordinate reorders PLAID routing and modestly broadens the pre-pruning candidate union, but this sample shows no useful MRR gain and only a sub-gate final-recall gain. This is ordinary query routing freedom, not Mirror-specific capacity.

**Hypothesis:** A task code can still be useful where a native query transformation cannot express the desired routing or can be fused more cheaply into an existing index path. This experiment provides no evidence for that case.

## C — Strongest counter-hypothesis

The entire effect is a conventional orthogonal query transform. Candidate-union recall rises because more relevant documents are exposed before pruning, while the resulting candidate expansion dilutes the first-stage ranking. No new physical index or independent logical retrieval function is demonstrated.

## U — Unconfirmed

- Full 171k-document TREC-COVID retrieval, another checkpoint, GPU PLAID, and other task/domain roles.
- Repeated, randomized latency calibration, production throughput, and stable tail-latency ordering.
- Whether a different non-orthogonal task coordinate can improve final quality after a same-cost native router control.
- The 20-query audit is small and the collection is deliberately bounded by positive qrels plus random background; no standard benchmark leaderboard claim follows.

## Protocol amendments and replay history

Before reading any audit metric, CPU profiling led to a deterministic 4,096-passage codebook sample and official-default `index_bsize=64`; the complete candidate collection remained indexed. Two first runs were interrupted before an index or metric was saved. The first complete development+audit pass then failed only at report serialization because of an archive-path typo; its audit output was not shown or persisted. A replay subsequently reached report serialization but exposed a frozen-development variable-scope bug, again before audit output. The corrected run reused the exact saved development freeze, ran the same audit arms, and saved query-level rankings. No observed audit metric was used to change a parameter.
