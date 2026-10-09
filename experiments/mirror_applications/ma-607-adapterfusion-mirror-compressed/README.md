# MA-607 — AdapterFusion over Mirror-compressed adapters

Status: SCREENING
Evidence lane: MECHANISM / STORAGE / COMPUTE
Base commit: `a5ba4d93` (worker-ready baseline plus MA-602–606 evidence)
Prior art: PA125 AdapterFusion

## Hypothesis

H: For source adapters whose rank-space operators lie on a shared orthogonal Givens orbit, storing one shared low-rank basis plus per-task Givens coordinates preserves AdapterFusion composition quality while reducing actual payload bytes versus independently stored adapters and generic shared-basis coefficient matrices.

## Mirror insertion

> **Mirror insertion:** this experiment adds a fifteen-angle task coordinate `m_t` to shared input/output low-rank adapter factors, reconstructing each source adapter's rank-space transform before the existing contextual fusion gate, so multiple logical source adapters can be fused without storing all task-specific factors.

AdapterFusion's fusion gate is held common across methods and trained after source adapter fitting. The teacher is intentionally aligned to the tested Givens orbit; the conclusion is limited to this structured case. A generic shared 6×6 coefficient matrix per task is the nearest ordinary shared-basis control.

## Gates

PASS requires fresh fusion NRMSE no worse than 1.10× the full independent adapter bank in each world, at least 25% fewer actual bytes than full adapters, and at least 10% fewer bytes than the generic shared-basis coefficient control. Otherwise FAIL. A pass establishes only aligned synthetic adapter compression, not general task composition.

## H / T / D / C / U

**H — hypothesis:** for source adapters on a shared orthogonal Givens orbit, one common rank-6 basis plus fifteen angles per source preserves AdapterFusion composition while reducing actual bytes versus independent adapters and a generic per-task 6×6 coefficient matrix.

**T — execution:** four source adapters in 32-dimensional input/output space, rank 6, intentionally generated from common A/B factors and per-task Givens rotations; AdapterFusion target is an input-conditioned softmax mixture over all four source outputs. Two development worlds (60701/60702), 2,048 source examples per task, 1,200 source-fit updates and 800 fusion-gate updates, then 1,024 held-out examples. Compared independent rank-6 source adapters, shared A/B plus unrestricted 6×6 per-task matrices, Mirror angles, and oracle. All methods evaluate four adapter outputs. Fresh 60711/60712 stayed sealed because the byte gate versus the generic shared-basis control failed.

**D — FAIL under the frozen gate:** Mirror and controls preserve composition essentially exactly: Mirror normalized MSE 3.7753e-5 / 1.6635e-6, within numerical noise of full independent adapters and generic shared coefficients. Mirror payload is 4,761 B versus 8,989 B full adapters (47.0% lower), but only 7.5% below the ordinary shared-basis control at 5,145 B, short of the required 10%. Mirror compute proxy is 1,896 versus shared control 1,680 (12.9% higher). Measured Mirror source fitting took 10.1–10.8 s versus 1.1–1.2 s for the generic shared control. Fresh remained sealed.

**C — strongest counter-hypothesis:** the ordinary shared low-rank basis recovers the same functions with almost the same bytes and less transform compute/training time. The measured Mirror storage increment is too small to compensate for its slower transform optimization and inference proxy.

**U — boundaries:** teacher adapters were intentionally aligned to the Givens orbit, making this a favorable structured case. No arbitrary adapters, natural task checkpoints, or language models were tested.

## Facts / interpretation / hypothesis

- **Fact:** all eight saved payloads match byte length and SHA-256; reload error is zero; three tests pass.
- **Interpretation:** Mirror codes can replace nearly half the bytes of four independent adapters on an aligned function family while retaining fusion quality, but the stronger general shared-basis control comes within 7.5% of the bytes and uses less compute. Strict Mirror-specific gate therefore fails.
- **Hypothesis:** low-dimensional orthogonal task orbits can provide useful adapter-bank compression when the alternative is independent storage; additional benefit over ordinary shared-basis coefficient codes is small and may not survive runtime costs.
