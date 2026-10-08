# MA-391 — Quotient/remainder Mirror compositional embeddings

Status: **FAIL — no Mirror-specific Pareto advantage over elementwise multiplication.**
Evidence lane: QUALITY / STORAGE / UNIQUENESS / RUNTIME
Base commit: `bec8819`
Prior art: PA59, complementary quotient/remainder embeddings.

## H — hypothesis

For separable quotient/remainder labels, fixed additive complementary tables should be sufficient. For an XOR interaction target, one per-ID Mirror angle over those tables should beat addition by at least 0.05, remain within 0.02 of the best ordinary factorized operator, and use at most 80% of the independent-table bytes. Decoded-vector uniqueness is also measured.

## Mirror insertion

For ID `i`, `q=i//32` and `r=i%32`; the address pair is deterministic and unique. Fixed addition uses `Q[q]+R[r]`. Mirror adds one learned coordinate: `cos(theta_i)Q[q]+sin(theta_i)R[r]`. Controls also include elementwise multiplication and concatenation with a shared projection.

## T — protocol and execution

Protocol frozen before development. Two seeds (39101/39102), two target families (separable modular labels and low-bit XOR), vocabulary 1,024, 16D embeddings, 32 rows per factor table, eight classes, and 1,200 Adam updates per method. Five methods: independent full table, addition, multiplication, concatenation, and Mirror. Each method uses the same deterministic quotient/remainder addresses and task examples within a seed/family.

All reported quality and uniqueness are measured from reloaded FP16 payloads. Payload accounting includes full or factor tables, per-ID angles, projection, classifier, metadata and ZIP/NPY overhead.

## D — decision

**Fact:** XOR test accuracy for full/add/multiply/concat/Mirror was 1.000/0.159/1.000/0.157/0.996 in seed 39101 and 1.000/0.155/1.000/0.136/1.000 in seed 39102. Mirror was within 0.02 of the strongest factorized control (multiplication), used 5,790 B versus 34,014 B full, and produced 1,024 distinct decoded vectors from 1,024 unique address pairs. However, multiplication matched full-table accuracy with 3,506 B and lower NLL (0.0004–0.0005 vs Mirror 0.106–0.108), so Mirror cost 65% more bytes without improving accuracy. On the separable family, additive accuracy was only 0.145–0.164, while multiplication and Mirror were 1.0. Thus the preregistered Mirror-vs-additive noninferiority clause fails literally in the direction of Mirror being much better, indicating that the additive control did not learn its expected target under this fixed budget. Twenty payloads replayed accuracy, NLL, and decoded uniqueness exactly; four tests passed. Fresh seeds 39111–39113 remain unopened.

Compute: Mirror and multiplication each use two factor lookups; Mirror uses a 32-MAC/query operator proxy and multiplication 16, while addition uses 16. Test throughput is recorded in `RESULTS_CORE.csv` as a batched local CPU harness metric, not as optimized serving latency. On the XOR family, Mirror training took 1.56/1.24s, multiplication 1.41/1.31s.

**Interpretation:** The angle code creates a unique decoded vector for each logical ID using a small physical table, and it solves the registered XOR task. Yet a simpler multiplication composition reaches the same quality with less storage and lower NLL. The result does not establish a Mirror-specific advantage. The fixed-addition target exposed an optimizer/control limitation rather than evidence that Mirror should replace addition generally.

**Hypothesis:** Mirror angle coordinates may be useful when a task demands a per-ID choice between factor vectors, but in this task the ordinary elementwise product already captures the interaction efficiently.

## C — strongest counter-hypothesis

The modular additive target is representable by complementary tables, but the fixed 1,200-update linear training run stayed near chance for addition and concatenation. This suggests an optimization/initialization issue in those ordinary controls; no capacity limit is inferred from their low accuracy.

## U — unresolved

Natural recommendation or language tasks, longer training for the additive control, larger partition sizes, frequency skew, efficient kernels, and fixed-byte convergence frontiers remain untested. No natural embedding or capacity claim is made.
