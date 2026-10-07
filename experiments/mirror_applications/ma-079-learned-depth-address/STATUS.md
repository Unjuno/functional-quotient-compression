# MA-079 status

- Status: PROMISING
- Branch: `research/ma-079-learned-depth-address-20261007`
- Base commit: `a761cc84422c323e78ad8290a2ff6581bc39723a`
- Last verified commit: pending
- Development complete: yes; seeds 79001–79002
- Fresh/audit opened: yes; seeds 79011–79013 after development gate
- Results committed: pending
- Verification committed: pending
- Registry row updated: pending

## H / T / D / C / U

- **H:** A compact polynomial depth address can use one shared block to recover held-out intermediate layer functions from sparse depth supervision, at lower payload and better held-out quality than simple shared controls.
- **T:** Eight 8×8 linear maps; only depth indices 0, 2, 5, 7 were supervised. Compared tied, static rank-1 LoRA, generated diagonal gain, Mirror polynomial Givens address, and sparse-trained untied controls. Two development and three fresh worlds; aligned and independent teacher conditions; 600 Adam updates at LR 0.01.
- **D:** PROMISING for the aligned interpolation mechanism: Mirror held-out MSE was near zero in all 3 fresh worlds, lower than tying and generated gain, with a 2,149B payload. Independent layer targets were not recovered.
- **C:** The teacher exactly follows the same polynomial Givens trajectory used by Mirror. Also, untied weights at unseen depths had no training targets, so their held-out error is not an independent-capacity upper bound.
- **U:** Fully supervised untied quality, nonlinear Transformer blocks, natural-language NLL, extrapolation to unobserved depth ranges, and optimized runtime.

**FACT:** 50 rows replayed; exact payload bytes; max metric delta 4.94e-10. Fresh aligned held-out median MSE 3.36e-12 vs 0.0646 generated gain, 0.0690 tied; 2,149B vs 3,753B sparse-trained untied. Mirror eager CPU wall was 4.5x generated gain. Tests 2/2 pass.

**INTERPRETATION:** A low-dimensional depth coordinate can interpolate multiple logical functions from sparse depth labels when the functions follow its trajectory. The byte result is measured; no capacity advantage over fully supervised independent weights is established.

**HYPOTHESIS:** Structured depth addresses may reduce per-layer supervision and storage; irregular functions will need private residuals or independent blocks.

## Next action

Update the registry and program trackers, commit and push this result, then continue to MA-086.

## Blockers

None.

## Decisions / rulings

The untied control received only sparse-depth targets, matching the fixed supervision budget. Its held-out quality is reported transparently as a sparse-trained control, not an all-depth oracle.
