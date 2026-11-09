# MA-690 status

- Status: FAIL (Mirror-specific attribution); local shared/private operator frontier measured
- Branch: `research/ma-690-orbit-private-frontier-20261109`
- Base commit: `53040b26`
- Protocol frozen before first run; two documented pre-audit accounting/implementation amendments
- Development: complete; seeds 69001/69002
- Fresh/audit: sealed after exact native-family alias
- Results committed: pending
- Verification: 5 tests passed; full-residual replay maximum error <2.3e-8

## H — Hypothesis

Across target families, a shared View coordinate plus low-rank private residual may beat a residual-only bank at matched actual serialized bytes.

## T — Treatment and controls

16x16 shared backbone, 8 tasks, 7 target generators, 6 candidate coordinate families, alpha amplitude values 0/.25/.5/.75/1, residual ranks 0/1/2/4/8/full, and 2 development seeds. Cross-evaluated every candidate on every generator; compared independent full matrices, native family coordinates, and byte-matched residual-only SVD. Oracle least squares/SVD; zero training updates.

## D — Decision

FAIL for Mirror-specific value: all 2,520 candidate/native pairs have identical function, compute, bytes and serialized hash. The local shared/private representation frontier is informative: 64/2,520 points pass error<=.02, bytes<=75% independent and compute<=2x, all at alpha=0. In same-family alpha=0 rank0 settings, payload is 39.8–48.9% of independent with NRMSE<=5.8e-8. No alpha>0 point passes the .02 quality and 75% byte gate; full residual is exact but larger than independent. Family code plus residual beats the same-byte best residual-only SVD in 587/2,520 grid points, across multiple families/generators, while native controls exactly alias the representation. Fresh remains sealed.

## C — Strongest counter-hypothesis

Native IA3/BatchEnsemble/VeRA/low-rank/Givens/BOFT local parameters already provide the same functions as the View coordinates; residual allocation explains heterogeneity.

## U — Unconfirmed

No learned adaptation, language quality, nonlinear orthogonal finetuning, downstream generalization or optimized accelerator kernels.

## Next action

Integrate MA-690 FAIL and frontier findings into the registry/claim/status board; select the next eligible untested P0.
