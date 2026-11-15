# MA-1040 — ScaNN anisotropic metric task View

**Decision: FAIL** for the registered Mirror-specific task-metric hypothesis on this bounded official-model/ArguAna CPU screen.

## H — Hypothesis

Two development-selected query-length-role Givens coordinates would improve fresh ArgUana nDCG@10 and Recall@10 by at least 0.5 percentage points over one fixed ScaNN AVQ index, while sharing the index and staying within the p95 gate. A Mirror-specific result must additionally beat a matched ordinary two-coordinate linear transform by at least 0.2 pp nDCG@10.

## T — Test

- **Method:** ScaNN 1.4.2 CPU, dot-product AVQ tree, 128 leaves / 16 searched, 2-D asymmetric-hashing blocks, anisotropic threshold 0.2 and 100-neighbor reorder. The same serialized searcher serves every arm.
- **Embeddings:** pinned Nomic Embed v1.5 INT8 ONNX snapshot, `e9b6763023c676ca8431644204f50c2b100d9aab`, masked mean, 768-D layer norm and L2 normalization. After an initial full-length attempt was stopped before any outputs or metrics were saved, AM-01 capped the input at 256 tokens and added resumable batch checkpoints; model revision, benchmark, index settings and gates did not change. This protocol change preceded all dev metrics and the audit.
- **Data:** official BEIR ArgUana, 8,674 corpus documents and 1,406 qrel queries. A salted qid SHA256 split, independent of qrels, assigns 701 development and 705 audit queries. Query-length roles use the median development wordpiece length, 223; equality is short. All corpus documents are in the shared index.
- **Development:** 13 fixed Givens angles per role, zero optimizer updates. The selected codes are short `+0.30 rad`, long `+0.15 rad`. Development freeze and code were committed before opening audit qrels.
- **Compute:** 8,674 document encodings took 1,034.19 s and 1,406 query encodings 191.99 s on CPU (four ONNX threads); ScaNN index build took 3.107 s. ScaNN does not expose comparable SIMD/FLOP counters through this API; index/search config and latency are reported instead.
- **Runtime:** five randomized arm-order repetitions for every audit query (3,525 observations per arm), including the dynamic query transform and single-query ScaNN search, excluding tokenization/encoding.

## D — Results: FAIL

| Audit arm | nDCG@10 | Recall@10 | MRR@10 | overlap Recall@10 vs exact cosine | p50 / p95 ms | Complete bytes |
|---|---:|---:|---:|---:|---:|---:|
| Native ScaNN | 0.3519684 | 0.7333333 | 0.2317916 | 0.9717730 | 0.0997 / 0.4413 | 169,461,177 |
| Mirror role Givens | 0.3520188 | 0.7347518 | 0.2314798 | 0.9643972 | 0.1119 / 0.4496 | 169,461,272 |
| Matched native linear transform | 0.3520188 | 0.7347518 | 0.2314798 | 0.9643972 | 0.1109 / 0.4566 | 169,461,431 |

Mirror versus native changes nDCG by **+0.0050 pp**, Recall@10 by **+0.1418 pp**, and MRR by **−0.0312 pp**. Both quality gains miss the registered +0.5 pp gate. A 10,000-replicate paired bootstrap gives 95% percentile intervals of [−0.1073, +0.1317] pp for nDCG, [0.0000, +0.4255] pp for Recall@10 and [−0.1465, +0.0791] pp for MRR. The bootstrap is descriptive; the frozen gate uses the point metrics.

The matched 2×2 native linear query transform produces **the exact same top-10 document IDs** as Mirror on all 705 audit queries and exactly the same metrics. This is expected mathematically: a Givens rotation is an ordinary constrained linear map on the two selected query coordinates. Mirror is 95 B smaller than the explicitly serialized four-coefficient-per-role linear control, but this is a compact encoding of the same rotation; an ordinary angle parameterization can use the same bytes. It does not establish a Mirror-specific quality or capacity gain.

The measured Mirror p95 is 1.88% above native, within the preregistered +10% latency ceiling; p50 is 12.2% above native. The linear control p95 is 3.46% above native. The paired bootstrap intervals include zero for nDCG and MRR; recall’s upper interval is +0.426 pp, still below the +0.5 pp gate. These are repeated randomized CPU measurements on one host and one index; they do not imply production throughput.

## Actual storage and compute

The single shared ScaNN index serializes to **30,889,586 B**, including its quantized dataset, reordering data, AH codebook, partitioner and config. The pinned Nomic model/config/tokenizer bundle is **138,243,620 B**; the required document-ID map is **327,971 B**. Native total is **169,461,177 B**. Mirror adds an actual **95 B** role-angle/config payload for **169,461,272 B** total. The full-matrix native control config is 254 B, for **169,461,431 B**. The index is shared and charged once; the temporary embedding arrays and query texts are not inference state. Every serialized index file and model asset hash is recorded in `runs/VERIFICATION_REPLAY.json`.

Embedding cost is 1,226.18 CPU seconds total; index construction 3.107 s. No trainable parameter updates occurred. Active ScaNN operation counts are not exposed; the fixed searchable structure uses 16/128 leaves, 384 two-dimensional AH blocks per vector, and top-100 reorder before returning 10 results.

## Fact / Interpretation / Hypothesis

**Fact:** On 705 hash-held-out ArgUana qids, Mirror nDCG@10 is 0.3520188 and Recall@10 is 0.7347518; both are below their registered absolute-gain gates. MRR is slightly lower than native. The matched linear control is ranking-identical. Complete payloads are 169,461,272 B Mirror and 169,461,177 B native. Five interleaved latency repetitions per query were measured.

**Interpretation:** A role-conditioned query metric can change ScaNN's approximate ranking while using one physical index, but the measured gains are far below the frozen usefulness threshold and are fully reproduced by conventional linear query transformation. The modest role-code footprint is not a useful Pareto improvement in this test.

**Hypothesis:** A query code with a richer, non-isometric metric family or cross-dataset task address might use the shared ScaNN index differently; this run offers no evidence that such a code would beat its matched native control.

## C — Strongest counter-hypothesis

The entire functional change is ordinary query-side linear metric adaptation. The matched control reproduces every audit ranking exactly; any small metric movement is attributable to query transformation and finite audit composition, not Mirror-specific freedom.

## U — Unconfirmed

Other ANN datasets and query domains, a second encoder, larger/richer metric codes, ScaNN GPU or production-scale performance, longer than 256-token inputs, and whether an independently selected task-specific anisotropic quantization/index code can outperform native ScaNN without cloning its index.

## Verification

`source/verify_results.py` replays split integrity, development angle selection, all audit quality metrics, serialized payload arithmetic and file hashes. It also reports paired bootstrap intervals. Tests exercise Givens/linear exact alias, norm preservation and split disjointness. See `VERIFICATION.json` and `runs/VERIFICATION_REPLAY.json`.
