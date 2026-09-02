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

## Boundary

These are fixed-probe research claims, not benchmark or production-quality claims. They do not establish whole-model compression quality, additivity of component distortions, or 64x feasibility/impossibility.
