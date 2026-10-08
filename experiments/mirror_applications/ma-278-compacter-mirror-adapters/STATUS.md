# MA-278 status

**FAIL — development access gate missed; fresh worlds sealed.**

Integrated from `research/ma-278-compacter-mirror-adapters-20261008` (source result `c159672`; verification provenance `1d93af7`). Local model tests and the preregistered gate audit were rerun successfully.

## H / T / D / C / U

- **H:** A compact task-specific Givens coordinate over shared Compacter-style Kronecker slow atoms and shared rank-one fast factors will recover independently varying adapters at lower payload than native Compacter and beat scalar coefficient modulation.
- **T:** Six 16x16 linear adapters; rank-2 Kronecker sum; 8192 train and 2048 held-out inputs per task/world; Adam, 500 updates, batch 128; worlds 27800 and 27801; LRs .003 and .01. Controls: tied, native Compacter (task-local rank-one fast factors), scalar shared-factor coefficients, Mirror rotation coordinates, independent full adapters.
- **D:** FAIL at screening. Mirror MSE is 0.0339–0.0665 across the two development worlds and two LRs, compared with native Compacter 5.4e-8–3.7e-3 and independent 3.7e-6–1.4e-5. Mirror ties scalar control on serialized bytes (2525 B) but is worse in MSE; it is slightly slower than Compacter and scalar. Fresh worlds 27802–27804 remain unopened because the preregistered access gate failed.
- **C:** The target's task-specific rank-one fast factors vary independently. Per-task rotations of shared fast factors cannot represent those variations; the result may reflect deliberate misalignment with the proposed Mirror coordinate.
- **U:** Aligned feasibility is not tested. No capacity, language-model, optimized-kernel, or GPU claim. Fresh generalization is not established.

## Fact / Interpretation / Hypothesis

- **Fact:** The Mirror parameterization failed the development gate at both learning rates and in both development worlds. Actual payload is 2525 B for Mirror/scalar, 2593 B Compacter, 7849 B independent.
- **Interpretation:** In this natural independent-factor task, Mirror rotation has no observed value over scalar modulation and is less expressive than native Compacter.
- **Hypothesis:** An explicitly aligned task family may be recoverable, but it would establish mechanism feasibility only; it would not reverse this negative result for unaligned task-local factors.

Fresh worlds were kept sealed. This result is a fixed-budget screen, not a near-convergence capacity comparison.
