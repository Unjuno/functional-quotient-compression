# MA-503 status

- Status: **FAIL** (held-out quality/storage gates; no Mirror-specific attribution)
- Branch: `research/ma-503-factorized-layer-task-reft-20261008`
- Protocol frozen before development: yes (`freeze.json`)
- Amendment 1: cache-only source correction, documented and frozen before canonical rerun
- Development: complete (50301/50302 rerun under amendment 1)
- Fresh/audit opened: **no** (50311–50313 remain sealed)

## Decision

At rho=0, heldout error differed sharply by seed (0.00215 vs 6.75411), and the factor payload was 2,568 B versus 2,685 B for the dense pair code table, missing the <=0.50x gate. Native bilinear factors exactly alias Mirror. At rho=.25, heldout error rose to 0.563/22.515. The full-matrix oracle is valid; fresh stays sealed.

## Fact / interpretation / hypothesis

- Fact: native bilinear and Mirror have exact payload/output aliases in both worlds and both rho settings.
- Fact: the factor model produced eight distinct heldout functions; rho=0 heldout quality did not replicate across the two worlds.
- Interpretation: the direct result fails both the generalization robustness and actual-byte ratio thresholds; current coordinates are ordinary factorized parameters.
- Hypothesis: checkerboard matrix completion may be underidentified for this factor rank; the frozen run did not isolate identifiability from optimizer initialization.
