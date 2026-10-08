# MA-1075 — HSTU retrieval with interest and candidate-role Views

Status: FAIL (development screening)  
Base commit: `16807a7d6dc17a834a44ed3cc793ccc28b615691`  
Random draw #17: pool size 1040, index 999; candidate MA-1075.

## Hypothesis

H: A compact per-role Mirror coordinate on the HSTU session representation improves held-out genre-conditioned retrieval beyond native HSTU shared scoring and an equal-byte standard low-rank role scorer.

## Mirror insertion

> **Mirror insertion:** this experiment adds `m` to the late HSTU session/user scoring interface so that candidate-role-specific retrieval functions can be expressed without duplicating the HSTU encoder or shared item embedding substrate.

One official HSTU session encoder and one shared item embedding table are physical state. `m` is a compact per-genre View on the session vector used during candidate scoring. Logical views are scores for multiple movie-genre candidate roles.

## Prior art and controls

PA342 establishes HSTU sequential recommendation; PA341 establishes VQ-Rec transferable item codes. Meta's public `generative-recommenders` implements HSTU. VQ-Rec is not reproduced in this experiment; results will be explicitly scoped as an HSTU retrieval screen and cannot claim superiority over VQ-Rec. Controls are native HSTU dot-product scoring, an ordinary pairwise per-dimension genre-gain control with the same code shape, and a per-genre dense upper reference when feasible.

## Frozen split and gate

MovieLens-1M, user histories sorted chronologically, first 80% train, next 10% dev, last 10% fresh, minimum 10 events/user. Hyperparameters are selected on development only. Primary metric is mean per-genre nDCG@10 over next-item retrieval. PASS requires fresh improvement over native HSTU and same-size pair-gain control with actual serialized bytes and CPU P99 reported and no >10% P99 regression. FAIL if the ordinary control matches or beats Mirror at similar bytes or Mirror misses native quality without a Pareto gain.

## Runtime / provenance

CPU-only screen. HSTU source: Meta `generative-recommenders`, commit `25e032d6f1e29b7f7652aa2b0a9f75499a4b1940`. FBGEMM jagged layout utilities are provided by deterministic Python equivalents in the experiment adapter. No changes to the external checkout or nanoGPT.

## Results (development screen only)

FACT: On the corrected chronological split, native HSTU reached macro genre nDCG@10 0.119023; pair-gain 0.117436; Givens Mirror 0.117178. Each condition saw 16,000 training examples and 250 updates at seed 41075. The official HSTU architecture used 2 blocks, 1 head, width 32 and maximum history 50.

FACT: Actual serialized inference payloads (HSTU weights, item table, item ID map, genre map, role code and config) were native 826,846 B; pair-gain 828,880 B; Mirror 828,186 B. The one-request HSTU cache plus item history IDs was 47,952 B for all methods. All three methods have equal 39,969,792,000 estimated training MACs under the stated upper-bound proxy; training wall times were 9.54–10.00 s.

FACT: Single-threaded interleaved batch-1 P99 for full HSTU encode plus all-item scoring was native 1.135 ms (881 QPS), pair-gain 1.080 ms (926 QPS), Mirror 1.207 ms (829 QPS).

INTERPRETATION: Mirror is dominated by native HSTU on dev quality, serialized bytes and P99. It also has lower dev nDCG than the same-code-count pair-gain control and 11.7% worse P99. This fails the registered development screening gate.

HYPOTHESIS: Role-specific Givens views do not add useful retrieval function for this HSTU/MovieLens screen at the measured fixed budget. This is not a capacity or general recommendation claim.

U: No fresh temporal metrics, seed replication, near-convergence run, or native VQ-Rec comparison. VQ-Rec (PA341) remains the strongest untested item-code counter-hypothesis. The result therefore says nothing about superiority to VQ-Rec.

The older untied-timestamp development run is retained at `source/development_seed41075_pre_boundary_amendment.json` and excluded from this decision. The fresh split was used only for boundary-integrity checking; no fresh quality metric was computed.
