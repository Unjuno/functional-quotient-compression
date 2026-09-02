# Root Pricing Claim Boundaries

## FQC-C015 — Master optimality is not candidate-family completeness

**Status:** ACTIVE  
**Evidence lane:** THEORY / finite-family optimization  
**Origin:** D64–D65

Exact optimization of the assignment master over a declared root pool does not by itself certify that the root candidate family is complete. These are separate proof obligations.

## FQC-C016 — Single-column stopping can fail

**Status:** ACTIVE  
**Evidence lane:** EXACT_TOY  
**Origin:** D65 complementarity counterexample

With shared prerequisites, absence of an improving feasible single omitted root does not imply absence of an improving omitted root coalition. In the D65 two-block example, A and B are individually infeasible at 11 bits but jointly feasible at 8 bits because they share a 4-bit prerequisite.

These claims should be merged into the machine-readable ledger after the current reconstruction PR is stable.
