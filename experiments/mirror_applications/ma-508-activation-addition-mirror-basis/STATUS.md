# MA-508 status

- Status: FAIL for Mirror-specific byte gate; generic shared-basis compression works on the synthetic orbit
- Branch: `research/ma-508-activation-addition-mirror-basis-20261009`
- Base commit: `f85b2963`
- Protocol frozen: yes; A1 basis/seed clarification committed before development
- Development complete: yes; deterministic encoder validation, 2 worlds × 3 seeds
- Fresh/audit opened: yes; 3 worlds × 3 seeds × 2 residual regimes × 5 methods
- Results committed: pending
- Verification committed: pending
- Registry row updated: pending

## Decision

**H:** Mirror activation-addition Views compress aligned behavior vectors beyond generic basis coefficients.

**T:** Synthetic 64D rank-4 activation steering, 64 behaviors, fixed shared basis/seed, explicit vectors, FP32/FP16 generic coefficients, FP16 Mirror angles, and private residual fallback.

**D:** FAIL for the Mirror-specific gate. At `rho=0`, Mirror used 3,361B / .000385 NRMSE versus generic FP16 coefficients at 3,365B / .000213. Only 4B were saved, far below the registered 20% margin; eager CPU decode was ~11x slower. At `rho=.1`, Mirror error was .141; storing per-behavior private residuals restored .000361 but raised payload to 4,061B.

**C:** The generic FP16 coefficient table already represents the aligned rank-4 family with nearly minimum per-behavior state. Mirror's format does not reduce it materially.

**U:** Natural/pretrained behavior vectors, selective private allocation and optimized inference kernels remain untested.

## Next action

Commit results and verification, update registry/claim/status board, run integrity verification, then continue to MA-510/511.

## Blockers

None.

## Decisions / rulings

A1 made the shared physical basis/seed identical across development and fresh bank worlds and explicitly charged it in every relevant payload. No fresh setting was changed.
