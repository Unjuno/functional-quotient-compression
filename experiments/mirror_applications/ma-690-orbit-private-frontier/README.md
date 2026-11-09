# MA-690 — Orbit/private frontier across View families

Status: **FAIL for Mirror-specific value; scoped local operator frontier measured**  
Evidence lane: MECHANISM / OPERATOR-SPACE / ACTUAL BYTES / COMPUTE  
Protocol initially frozen at `8f2c9ccc`; pre-run and implementation/accounting amendments are documented in `PROTOCOL.json`.

## Hypothesis

H: Across multiple target generators, one shared matrix plus a compact family View code and a low-rank private residual can reach lower normalized operator error at lower complete serialized bytes than the best residual-only SVD bank at the same byte budget.

## Mirror insertion

> **Mirror insertion:** this experiment adds a per-task coordinate `m` over a shared family-specific operator basis around a common matrix `W`, so that multiple logical task matrices can be represented without storing every full matrix independently.

The logical functions are eight task-specific linear operators. The native control stores the same direct family parameters that define `m`; this tests Mirror-specific attribution as well as the shared/private frontier. Givens and BOFT use local tangent bases, not complete nonlinear transforms.

## T — Treatment and controls

One 16x16 backbone; 8 tasks; development seeds 69001/69002; seven target generators (IA3, BatchEnsemble, VeRA, low-rank, Givens, BOFT, and family-neutral); six candidate families; alpha amplitude mix 0/.25/.5/.75/1; private ranks 0/1/2/4/8/full. Every candidate was evaluated on every target bank. Controls were independent full matrices, native family coordinates, and residual-only SVD chosen by actual payload bytes. Coordinates were fitted by least squares and residuals by truncated SVD, with zero optimizer updates.

## D — Decision

**FAIL for Mirror-specific value.**

### Facts

- The corrected run contains 7,700 result rows and 2,520 family-code payload measurements across 6 candidates × 7 generators × 5 alpha values × 6 residual ranks × 2 seeds. The shared structured union had rank 71 of 256 matrix dimensions in both seeds.
- At alpha=0, same-family rank-0 views reconstructed with NRMSE at most `5.8e-8`. Their full payloads were 3,287–4,032B, or 39.8–48.9% of the 8,241–8,264B independent target banks. The amortized compute proxy was 264–320 MACs/query versus 256 for the dense apply baseline.
- Across the full candidate grid, 64/2,520 points met all preregistered quality, byte and compute gates. All were alpha=0. No alpha>0 point met the .02 error plus 75%-of-independent-byte gate. Rank-8 left about .09–.10 NRMSE already at alpha=.25; full residual restored error below `2.3e-8` but used more bytes than independent full matrices.
- At 587/2,520 grid points, family code plus residual had lower operator error than the best residual-only SVD payload that fit within the candidate's actual serialized bytes. The low-rank family showed this across multiple target generators and alpha values 0–.75 in both seeds.
- All 2,520 Mirror/native rows had the same function, compute proxy, byte count and payload hash under the canonical shared representation. Fresh seeds 69011/69012/69013 remained sealed.
- Verification passed 4 tests; maximum full-residual replay error was below `2.3e-8`.

### Interpretation

The shared family coordinates produce a useful storage point when the task updates lie in the chosen local structured span. Small private mixtures rapidly require more residual rank; by the measured .02 quality threshold, only the pure structured endpoint retains the 75% byte gate. A structured family code can beat a residual-only code at the same bytes, but the same gains are exactly represented by its ordinary native parameters.

### C — Strongest counter-hypothesis

IA3, BatchEnsemble, VeRA, low-rank coordinates, and orthogonal tangent parameters are ordinary native parameterizations of the same local operator bases. The byte/quality frontier is explained by these methods plus residual allocation; `m` contributes no extra functions beyond them.

## U — Unconfirmed

This is an oracle synthetic operator-bank screen, not learned adaptation. It does not establish downstream task quality, training sample efficiency, generalization, nonlinear Givens/BOFT behavior, optimized GPU kernels, or practical base-model amortization. Alpha is an amplitude coordinate, not a private-energy fraction.

## Preserved attempts

The first run double-counted the shared view in predictions; it is preserved under `source/attempts/initial_double_counted_view/` and excluded from conclusions. A later corrected run still serialized unused view coordinates at alpha=1; it is preserved under `source/attempts/pre_alpha1_code_pruning/`. The final run omits those unused coordinates. No fresh/audit data were accessed during these corrections.

See [PROTOCOL.json](PROTOCOL.json), [RESULTS_CORE.csv](RESULTS_CORE.csv) and [VERIFICATION.json](VERIFICATION.json).
