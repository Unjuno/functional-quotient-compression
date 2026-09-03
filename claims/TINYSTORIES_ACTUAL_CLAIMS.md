# TinyStories actual-model claim addendum

These claims are scoped to the user-supplied TinyStories GPT-Neo scale series (`1M`, `3M`, `8M`, `28M`) and the local fixed-probe experiments documented in `experiments/transformer/TINYSTORIES_SCALE_SERIES_ACTUAL_RESULTS.md`. They do not replace the main claim ledger and should be promoted there only after the next repository consolidation pass.

## FQC-TSA-001 — embedding geometry changes with model scale

**Evidence lane:** REAL_MODEL

Across the four supplied checkpoints, embedding share of floating checkpoint scalars falls from 85.86% (1M) to 49.50% (28M), while top-35 embedding explained variance falls from 0.79896 to 0.21571. Therefore a single scale-invariant embedding codec-selection rule is not supported by this series.

## FQC-TSA-002 — current 0.25-bps embedding family fails the tested functional gate

**Evidence lane:** REAL_MODEL

For the tested task-signature-root plus Fisher-ranked private-residual family, the 8M embedding gives six-probe mean KL 0.8349 at 0.25 bps and 0.3248 at 1 bps; 2 bps reduces mean KL to 0.0537. On 28M, richer calibration improves low-rate quality but the best tested 0.25-bps multi-probe configurations remain around KL 0.78-0.80. This is a negative result for the current codec family, not an impossibility result for all embedding codecs.

## FQC-TSA-003 — calibration breadth primarily improves private-bit allocation

**Evidence lane:** REAL_MODEL

At 28M, keeping the shared root fixed while widening the exception-importance calibration from one passage to five passages improves four-probe mean KL from 0.2741 to 0.2093 at 1 bps and from 0.1457 to 0.03875 at 2 bps. Rebuilding the root on the richer calibration has a much smaller and non-monotone effect. The dominant calibration benefit in this experiment is therefore the allocation of private exceptions.

## FQC-TSA-004 — shared-root/private-exception allocation is a discrete joint decision

**Evidence lane:** REAL_MODEL

At 28M and 0.25 bps with a rich exception selector, K=256 gives four-probe mean KL 0.7976, while K=512 gives 0.8917 and K=768 gives 0.9621 because larger roots leave fewer private exceptions. K=128 gives 0.8156. A single-probe experiment had instead preferred K=768, demonstrating that root complexity and private exception budget must be selected jointly and validated across multiple probes.

## FQC-TSA-005 — tied correction sharing has functional value in the tested 8M embedding codec

**Evidence lane:** REAL_MODEL

Using the same stored private residuals, applying corrections to both tied input-embedding and LM-head roles gives six-probe mean KL 0.8350. Applying them only to the LM head gives 2.6884, and applying input corrections only for the top 75% of stored exceptions by input-gradient importance gives 0.9218. Role-specific decoded state is therefore not automatically beneficial; in this tested representation the shared tied correction is functionally important.

## FQC-TSA-006 — additive individual task cost can select the wrong real-model compression set

**Evidence lane:** REAL_MODEL

For the 8M WQ K128 intervention family, all 256 layer subsets were evaluated on a fixed probe. The additive individual-KL model misses the actual oracle subset at cardinalities 2, 3, and 5. For three compressed WQ layers it selects `(1,2,3)` while the exact joint-KL oracle is `(0,1,3)`. On six unseen passages, the oracle triplet wins in 6/6 cases and reduces mean KL from 0.01745 to 0.01355 (about 22.3%).

Measured pairwise functional interactions almost completely repair the error: triple median prediction error falls from 14.25% to 0.55%, and the pairwise model recovers the exact oracle subset at every tested cardinality 2–7. Pairwise correction also transfers to WQ on 1M/3M/28M and to 8M WK/c_fc, though higher-order residuals are larger for the MLP.

## FQC-TSA-007 — task-priced codebook coalitions can outperform local codebooks at lower fixed-state count

**Evidence lane:** REAL_MODEL

For 8M Q and K roles, all 28 pair merge costs were measured on calibration data and all 105 perfect matchings into four pair-codebooks were enumerated. On six unseen passages:

- Q: task-priced4 mean KL 0.009606 vs local8 0.010366;
- K: task-priced4 mean KL 0.009713 vs local8 0.009862.

Thus the task-priced solution uses half as many codebooks while matching or improving holdout task quality. Naive contiguous four-codebook grouping does not achieve this, so coalition membership must be priced rather than inferred from layer adjacency.

## FQC-TSA-008 — WQ codebook-sharing quality improves with width in the tested scale series

**Evidence lane:** REAL_MODEL

Using the same task-priced four-codebook-vs-local-eight WQ procedure, holdout KL ratio `priced4/local8` is:

- 1M: 1.769;
- 3M: 1.276;
- 8M: 0.927;
- 28M: 0.909.

The ratio improves monotonically across these four checkpoints and crosses below one between 3M and 8M. This is evidence in favor of the positive-scaling hypothesis for shared representation in this architecture family, but it is not a universal scaling law and does not identify whether the mechanism is sample pooling, geometry change, fixed-state amortization, or a combination.

## FQC-TSA-009 — sub-bit task distortion and task-aware metrics are role-dependent

**Evidence lane:** REAL_MODEL

At 8M layer 2, increasing K from 16 to 256 monotonically improves weight NMSE for Q/K/V/O/c_fc, but task KL is not monotone for every role. Across three codebook seeds, V mean KL at K64/K128/K256 is 0.1163/0.1118/0.1219, while c_fc remains at very high KL despite improved reconstruction.

With the same K128 codebook and index rate, a diagonal-Fisher-like assignment metric improves holdout KL for Q and V but fails to transfer for K/O/c_fc under nested validation. Therefore reconstruction distortion and one universal task-weighting rule are both insufficient; the task metric itself must be role-aware and validated.

## Boundary

These are fixed-probe research claims, not benchmark or production-quality claims. They do not establish whole-model compression quality, additivity of component distortions, or 64x feasibility/impossibility. Centroid serialization precision and complete codec headers are not finalized in the VQ experiments, so codebook-count reductions are not yet emitted-byte certificates.
