# MA-011 — product-of-Mirror experts

Status: **FAIL** at the pre-fresh payload gate.  
Evidence lane: mechanism / storage / compute.  
Branch: `research/ma-011-product-mirror-experts-20261007`.  
Base commit: `b1ffe45e3812c12b2e05aa541f3c237401990804`.

## H — Hypothesis

Four logical nonlinear experts formed as pairwise products of two binary Mirror-view factor banks can preserve an aligned product-expert teacher using materially fewer actual bytes than four independent product experts.

## T — Test

Each role maps to a pair of binary factor addresses. The model applies two shared or role-specific linear factors, applies `tanh` to each branch, then multiplies the two vectors elementwise. The aligned teacher is generated from two shared matrices and Givens views; an independent teacher measures the private-function boundary. Controls are full independent experts, hard tying, rank-2 residuals, and scalar gates. Router labels are supplied and balanced.

Development uses world 110000 to choose one learning rate. Fresh worlds 110001–110003 remain locked until the development gate passes. Actual serialized inference payload bytes include both factor banks, all view/residual/gate tensors, and metadata.

## Development result

- Selected learning rate was 0.01 on development world 110000, pooled across five methods and both teacher modes.
- Aligned Mirror MSE was 9.44e-5 vs 6.77e-4 independent (0.140×); actual payload was 2,647B vs 3,684B (0.719×). The registered access gate required payload ≤0.65×, so fresh remained sealed despite passing the quality condition.
- Rank-2 residual used 3,606B and had 3.16× Mirror's MSE. Hard tying and scalar gates were smaller (2,141B and 2,331B) but had 9.59× and 9.05× Mirror's MSE. Eager CPU Mirror throughput was 2.63M examples/s vs 4.72M for rank-2 and 4.66M for independent.
- On independent teacher functions, Mirror MSE was 5.34e-3 vs 6.35e-11 for independent weights. The product factorization did not recover unrelated functions.
- Fresh worlds 110001–110003 remain sealed. These are development results only.

## D — Decision: FAIL

The development payload gate missed by 6.9 percentage points of the full independent payload ratio (0.719 vs maximum 0.65). This formal FAIL retains the promising development quality/control evidence but does not license a fresh quality claim. Fixed-update outcomes are not capacity evidence.

## Fact / interpretation / hypothesis

- **Fact:** aligned quality passed in development, actual payload ratio was 0.719 and missed the 0.65 gate; fresh was not accessed.
- **Interpretation:** the paired factor-address structure fit this aligned product teacher efficiently relative to simple controls, but failed the registered storage threshold and generalized poorly to independent factors.
- **Hypothesis:** larger product experts may amortize serialization overhead enough to improve the storage frontier; this requires a new registered candidate.

## C — Strongest counter-hypothesis

The aligned teacher is exactly generated from the same product-of-Givens structure. It may overstate the benefit for trained expert families. The registered payload cutoff is also sensitive to small-model serializer overhead.

## U — Unconfirmed

Fresh replication, larger matrices, nonlinear learned routers, deeper products, near-convergence capacity, optimized kernels, and language quality remain untested.
