# MA-767 — Online adaptation in functional-code space

Status: **FAIL** (scoped synthetic Mirror-specific claim)

## Fact

For 16 tasks generated from a shared 2D adaptation subspace in a 16D linear rule, fitting the two task-code coordinates from only four labeled examples per task yields relative test RMSE `1.31e-7`, with all 16 task codes retained. The serialized native low-rank adapter bank is 1,142 B vs 1,427 B for full per-task weights (20.0% fewer bytes). Code fitting uses 2,688 adaptation ops plus 512 ops to precompose all task weights; full fitting uses 81,920 ops. Direct dynamic code inference costs 409,600 MACs vs 131,072 for precomposed dense inference. At 8/16 examples, the code remains exact; full weights need 16 examples to recover exact output. Native low-rank adapters and Mirror codes have identical hashes, bytes, metrics and compute.

## Interpretation

Restricting adaptation to a known rank-2 task subspace can improve few-shot accuracy, reduce stored per-task state and cut closed-form adaptation work while retaining all prior tasks. Serving code dynamically costs more MACs, though precomposing task weights trades 512 total MACs for dense inference. Ordinary low-rank adapters fully explain the result; no Mirror-specific effect is established.

## Hypothesis result

**FAIL** for Mirror-specific value and the frozen byte gate: 20% byte savings miss the required 25%, and native low-rank adaptation is an exact alias. The adaptation-compute signal is useful but comes from an aligned synthetic family with a supplied basis.

## Frozen protocol / scope

PA198 motivates online task adaptation through compact functional state. Sixteen tasks use weights `w0 + B m` with input dimension 16 and rank-2 shared basis. Closed-form least squares fits `m` or a full vector from 4, 8 or 16 support examples/task. Test quality uses 512 heldout input examples/task, and all task states remain stored to measure retention. Compare no adaptation, full per-task weight copies, native low-rank adapters and Mirror. Full serialized bytes, adaptation examples/ops, dynamic and precomposed inference MACs are reported.

The task weights and shared basis are synthetic and supplied; this is not a trained self-organizing system, continual-learning benchmark or real online adaptation result.

## H / T / D / C / U

- **H:** Compact online task codes preserve test behavior while reducing bytes and adaptation cost relative to full rule updates and retaining old tasks.
- **T:** Seed 76701; 16D linear functions; 16 tasks; rank-2 shared basis; 4/8/16 supports; 512 test examples/task; closed-form solves.
- **D:** **FAIL** for Mirror specificity. At four supports, code RMSE is `1.31e-7`, uses 1,142 B vs 1,427 B full and 2,688 vs 81,920 adaptation ops; misses 25% byte gate and exactly aliases native low-rank adapters.
- **C:** Standard LoRA-like low-rank state on the supplied basis explains the entire gain.
- **U:** Learned basis, gradient adaptation, task distribution shift, realistic online environment feedback, retention under basis mismatch and deployed latency.
