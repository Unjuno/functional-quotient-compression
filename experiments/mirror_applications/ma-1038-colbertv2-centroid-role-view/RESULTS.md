# MA-1038 results — ColBERTv2 centroid role views

## H / T / D / C / U

### H — Hypothesis

Two query-length-role Givens views over one fixed residual-coded ColBERTv2 document-token index would improve fresh NFCorpus nDCG@10 by at least +0.5 absolute percentage points versus the same physical index without views, and by at least +0.5 pp versus the cheapest functional native control at comparable bytes and compute.

### T — Test

The frozen official ColBERTv2 checkpoint (`colbert-ir/colbertv2.0`, revision `c1e84128e85ef755c096a95bdb06b47793b13acf`; archive SHA256 `d07df4b2a9fe1848fbed333127a1a5b4f2d96106f420948f0db4450b018cb119`) encoded all 5,183 SciFact development documents and 3,633 NFCorpus audit documents to 128-D normalized token vectors. The official MaxSim form was evaluated exhaustively on CPU.

The physical compressed index used 256 centroids trained on a fixed 100,000-token sample from the SciFact corpus, one uint8 centroid ID and a 128-coordinate signed-int8 residual per valid document token, plus SciFact-derived per-coordinate scales. The View applies a block-diagonal rotation with one shared angle across all 64 coordinate pairs to each reconstructed document-token vector. The native control applies the inverse rotation to the query-token vectors. No encoder weights were trained; optimizer updates were zero.

Angles were selected using only 809 SciFact train-qrel queries over the complete SciFact corpus, from the frozen nine-point grid `[-0.20,-0.15,-0.10,-0.05,0,0.05,0.10,0.15,0.20]`. Query-role threshold was the SciFact median of 22 active tokens. Both short and long roles selected the boundary value +0.20 radians. The fresh audit was the official BEIR NFCorpus test qrels: 323 queries and 3,633 documents, with 322 short-role and one long-role query.

### D — Decision

**FAIL.** The view's nDCG@10 gain is +0.0007613, i.e. **+0.0761 pp**, versus the residual-coded no-view baseline. This misses the frozen +0.5 pp criterion. MRR@10 changes by −0.0010824, i.e. **−0.1082 pp**. The native query-side inverse Givens control has identical nDCG and MRR values and exactly the same serialized payload size as Mirror. Thus the observed ranking change is not Mirror-specific. The two selected role angles are equal, so there are not two distinct fitted role functions in this result.

### C — Strongest counter-hypothesis

Any small nDCG movement comes from a generic query-space orthogonal reparameterization and benchmark variation. The control is algebraically equivalent and reproduces the same metric values. The 322:1 audit role imbalance also means the long-role behavior is represented by only one qrel query.

### U — Unknowns

This does not establish results for PLAID's centroid-pruned traversal, other corpora, GPU inference, independently trained query encoders, or broad ColBERTv2 behavior. Search latency is NOT ESTABLISHED: repeated single-thread-order passes varied materially (full float 38.2–79.1 s, residual native 40.3–79.9 s, pushed-view control 23.1–74.7 s for 323 queries). The query-side transform is algebraically cheap, but no stable latency win is claimed.

## Fact

| Audit arm | nDCG@10 | MRR@10 | Actual serialized inference payload |
|---|---:|---:|---:|
| ColBERTv2 full-float document vectors | 0.3323742 | 0.5510860 | 726,467,488 B |
| Shared residual-coded index, no view | 0.3322895 | 0.5512322 | 511,690,535 B |
| Residual-coded bank + two stored role-view addresses | 0.3330509 | 0.5501499 | 511,690,756 B |
| Native query-side inverse Givens control | 0.3330509 | 0.5501499 | 511,690,756 B |

The full-float audit index has 561,120 valid document-token vectors, occupying 287,293,568 B as float32. The compressed token data costs 72,516,576 B for centroid codes, signed-int8 residuals, the 256×128 float32 centroid matrix and float32 coordinate scales. Document IDs, offsets, model, tokenizer and other reconstruction metadata are separately included in each complete bundle.

The shared model/tokenizer bundle is 439,104,784 B. Complete residual coding saves 214,776,953 B or 29.5646% versus the full-float bundle. The Mirror additions cost 221 B over the residual-coded no-view bundle (two float32 angle values, NumPy serialization header, threshold and metadata). Complete savings versus full float therefore remain 29.5645%, attributable to residual coding, not the role view.

The Mirror changes the top-10 list for 294/323 audit queries, with mean top-10 overlap 0.9297 against the no-view compressed index. Yet the query-side inverse rotation produces equivalent token dot products; over 4,096 fixed query/document pairs, the maximum absolute token-logit discrepancy was `7.1526e-7` (frozen tolerance `1e-6`). The native control's measured metrics and byte count are identical.

Each exhaustive audit arm scored 1,173,459 query-document pairs and used a 742,366,248,960 MaxSim MAC proxy. The algebraically pushed query-side Givens adds approximately 3,969,024 scalar FLOPs for the 10,336 query-token vectors. Applying the same rotation directly to every decoded document vector for each of 323 queries would add approximately 69,596,835,840 scalar FLOPs per audit pass. Optimizer updates: 0.

Measured per-arm full-MaxSim wall times varied across passes; full-float was 38.2–79.1 s, residual native 40.3–79.9 s and pushed view 23.1–74.7 s for 323 queries. Since arm order, cache state and CPU load were not independently randomized or stabilized, latency is NOT ESTABLISHED. The first end-to-end execution took 2,495.5 s; result replay took 157.9 s, final summary replay 261.7 s and independent verification 283.1 s, for 3,198.2 s total measured worker compute. The encoder-only stage time was not separately isolated; its time is included in the initial end-to-end wall figure.

## Interpretation

This screen demonstrates a substantial storage/quality point from ordinary centroid-plus-residual coding when the encoder bundle is included, with only a small nDCG loss relative to full-float vectors (−0.0085 pp). The role view changes rankings and has negligible address state, but fails the quality gate, yields no distinct role angles, and is exactly represented by a cheap native query-side transform. It therefore contributes no Mirror-specific frontier improvement here.

## Hypothesis for the next experiment

Centroid pruning and coarse candidate generation may expose useful task-specific views at the routing stage, but that must be tested independently against native PLAID pruning controls and actual index bytes. MA-1039 is the next candidate; no MA-1039 result is inferred from this experiment.

## Protocol and execution amendments

1. The first complete run finished all encoder, development and audit scoring, then stopped at summary serialization with a variable-name `NameError`. It had not printed or exposed audit metric values. The SciFact-selected choices (+0.20/+0.20, threshold 22) were recorded in `runs/DEV_FREEZE.json`, and the audit scoring/serialization was replayed from fixed vector caches without changing settings. A separate `verify_results.py` reran all three quality arms from fixed caches and reproduced every reported metric exactly (maximum difference 0); it also checked all component file sizes against the payload manifest.
2. Search timing was repeated incidentally during the result replay and independent verification. The materially variable timings are retained as negative evidence; no latency selection or claim was made.

## Boundaries

- One frozen official checkpoint; SciFact train qrels for selection; NFCorpus official test qrels for audit.
- Full CPU MaxSim only; not a PLAID implementation, production-scale index, or broad retriever result.
- Independent full query encoders were not run; the exact minimal native Givens control isolates the marginal View coordinate more directly.
- Large serialized model/index and intermediate vector arrays are kept outside Git. Their per-file byte lengths and SHA256 hashes are in `ARTIFACT_PROVENANCE.json`.
