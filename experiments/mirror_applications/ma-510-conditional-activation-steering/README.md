# MA-510 — Conditional activation steering with factorized condition and behavior Views

Status: **FAIL** (conditional quality gates and Mirror-specific attribution).
Branch: research/ma-510-conditional-activation-steering-20261008
Prior art: PA101, Conditional Activation Steering.
Protocol freeze commit: 36c60f0afec1a452b0304959c5e035994ffba444.

## H — Hypothesis

On two frozen synthetic contextual-activation worlds with 16 condition classes and 8 reusable behaviors, a rank-4 factorized condition/behavior Mirror will preserve conditional outputs (event recall >=.95, conditional behavior accuracy >=.95), hold hard-negative false-trigger rate <=.01, and use <=.65x the actual serialized bytes of explicit CAST condition and behavior vectors.

**Mirror insertion:** factor condition and behavior intervention vectors into two shared bases, and let a condition address choose a behavior code only when the condition score passes a calibrated gate. All basis, code, address, threshold and serialization state is paid.

The strongest attribution control is native PCA/SVD of those same banks. Explicit CAST is the native condition-vector similarity gate and behavior bank; independent condition-behavior vectors are the storage upper control.

## T — Executed protocol

The frozen protocol used dev seeds 51001/51002; 64 dimensions, 16 conditions, 8 behaviors, 128 support contexts per condition, separate calibration with 1,024 positives and 2,048 hard negatives, and evaluation with 4,096 positives and 8,192 hard negatives per world. Full fixed rank grid condition {2,4,8,16} × behavior {2,4,8}; primary point rank4/rank4. Calibration chose each method's threshold at <=1% calibration FPR. Actual uncompressed NPZ payloads include bases/vectors, all codes, addresses, threshold and schema. Three tests passed. Dev-only replay verified all 52 payload hashes and metrics; fresh seeds 51011–51013 remain unopened.

## D — FAIL

At rank4/rank4, Mirror payload was 4,334 B vs 7,532 B explicit CAST (0.575x); compute proxy 640 vs 1,088 operations/example. However, heldout event recall was .145/.071, conditional behavior accuracy .944/.830, and hard-negative false-trigger rate .0173/.0105 across the two seeds. All miss the frozen gates (recall >=.95, accuracy >=.95, FPR <=.01). Conditional-output relative RMSE was .934/.977.

Native PCA/SVD exactly matched Mirror payload sizes and output metrics at every rank pair and seed. Therefore rank-four byte/compute improvement is ordinary basis compression and cannot be attributed to Mirror. Fresh stayed sealed.

## C — Strongest counter-hypothesis

Hard negatives are activation mixtures near condition prototypes, and prototype cosine scores do not separate them from true condition contexts at the required recall/FPR point. The same router and intervention bases are standard CAST/PCA algebra. A natural LM activation distribution or a different condition representation could behave differently.

## U — Scope limits

No pretrained LM checkpoint is present in this checkout. This was a synthetic frozen-activation mechanism screen, not a natural-language or semantic CAST reproduction. It says nothing about prompt meaning, safety, human preference, or language-model quality. A future language claim needs a separately preregistered natural-prompt/frozen-LM experiment.

## Evidence classification

- **Facts:** rank curves, byte counts, timing, hashes, test outcomes and deterministic replay are in RESULTS_CORE.csv, runs/ and VERIFICATION.json.
- **Interpretation:** factorizing condition and behavior banks reduces synthetic payload and operation counts, but the task quality gate fails; native PCA fully explains the representation result.
- **Hypothesis:** learned semantic condition detectors on real model activations may improve false-trigger/recall tradeoffs, but these data do not test that possibility.
