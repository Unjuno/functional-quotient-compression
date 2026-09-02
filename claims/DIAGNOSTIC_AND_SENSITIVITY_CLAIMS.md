# Diagnostic and Functional-Sensitivity Claim Boundaries

## FQC-C017 — Shared-span residual is descriptive structure, not rate

**Status:** ACTIVE  
**Evidence lane:** THEORY / DIAGNOSTIC  
**Origin:** D58

For compatible same-shape operators, the singular spectrum of the stacked vectorized matrices gives the exact best rank-r shared linear-span residual by Eckart-Young. This does not by itself imply commutation, common support, low decoder complexity, or low serialized rate.

## FQC-C018 — Symmetric joint-diagonalization theorem has restricted scope

**Status:** ACTIVE  
**Evidence lane:** THEORY  
**Origin:** D58

Pairwise commutation is equivalent to exact simultaneous orthogonal diagonalization for a finite family of real symmetric matrices. The theorem must not be silently applied to arbitrary nonsymmetric RoPE/operator atoms.

## FQC-C019 — Fixed-input functional bounds are sufficient, not necessary

**Status:** ACTIVE  
**Evidence lane:** THEORY / SYNTHETIC_SANITY  
**Origin:** D61

The bilinear attention, rowwise-softmax, residual-output, composition, and final-logit inequalities are deterministic sufficient bounds on their declared inputs/domains. A large upper bound is non-decisive and does not prove task failure; direct replay remains necessary when the certificate is loose or outside the quality contract.
