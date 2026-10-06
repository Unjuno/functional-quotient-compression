# SM002 Conditional Virtual-Expert Controls Implementation Plan

Goal: test whether the SM001 gain survives inexpensive task-conditioned and partially shared expert controls, not merely Dense.
Architecture: extend the recovered SM001 finite-rule FFN pilot; keep router-free known task addresses, same input examples, one view per example. This is NOT a Transformer or a token-period experiment.
Tech stack: existing PyTorch 2.10.0+cpu, FP32 eager autograd, one thread/worker, deterministic algorithms. No GPU, quantization, compiled kernels, or dependency installation.
Spec: user request plus this preregistered experimental contract; prior approval was explicit to execute, so no repeated approval gate.

## Global constraints
- SM001 ZIP recovered byte-for-byte. Do not modify it. SHA256 ee9954b46cde4f6f1133c60790aa6be3cca8624b763cb3041e14ea0d612bcc7a.
- New isolated /mnt/data work directory; remote main checked as 0c3e5d78c9587ffb1ae7e8bf62b2ca4f9309fb61. No remote compute.
- Identical task information: one-hot task ID, rule ID and input value for ALL models. Deterministic task routing only; no label-derived routing.
- 4 tasks, 4 common rules, 32 input/output values, independent random permutation per private task-rule; full finite-domain retention audit, not unseen-label extrapolation.
- Frozen inference byte cap 44,155 B; canonical uncompressed FP32 SM02 includes architecture config, seed, shapes and paid learned scalars. No padding to manufacture equality. Maximum width that actually fits, bounded by 64. All candidate byte scans precede evaluation.
- Primary methods: Dense, assigned shear Mirror (rho=1), fixed positive direct gain after GELU (rho=0.5), learned per-task FiLM (identity initialization), per-task rank-2 residual linear experts at the first two layers (base ALSO trained; not original frozen-base LoRA), shared trunk + private output weight heads (shared output bias), private middle FFN layer + shared input/output layers, fully independent narrow experts.
- Controls: fixed-single-view shear and input-partition shear at private16/32, same-width independent full expert at private32 only (OUTSIDE byte cap).
- Same-width architectures copy the EXACT same common-only parent. Different-width architectures each use their dimension-matched common parent, trained 1,000 updates, lr=.003, batch128, AdamW wd=.01, clip1. Explain this unavoidable width/init difference. All allocated network weights are trained.
- Each primary method has equal development search: world92001/init171, private32/task, 4,000 updates, learning rates .001/.003/.01; select by common>=.99 first, then private accuracy, then NLL, then smaller lr. No audit-world selection.
- Freeze selected configs before generating audit worlds22001/22002/22003, initial seeds201/202/203. Private counts4/8/16/32. 4,000 updates with identical input stream and optimizer budget within each world/load (but development-selected lr may differ).
- Record learning curves every500 updates, model snapshots each checkpoint, all private/common accuracy and rule-level retention. Pretraining at zero-private is separate.
- Throughput/main panel may run on 3 workers: process CPU seconds and wall seconds both saved, neither claimed as isolated wall performance. Isolated sequential timing and actual elapsed-time stopped experiments performed after workers exit.
- Actual equal wall TRAINING-LOOP time follow-up at private32 only: a model-neutral wall budget is fixed after development-only calibration, before any audit training. All primary methods use that exact budget sequentially; exclude data construction, eval, serialization; include forward/backward, AdamW, clipping. Include count/overshoot. This is online-optimizer elapsed time, not end-to-end data-pipeline time or equal FLOPs.
- Budget and quality claims use achieved data, never inferred 6,079/4,000 speed ratio (that earlier ratio was set using timing, not independent evidence).

## H/T/D/C/U
H: Mirror's task-addressed nonlinear conjugation has a useful advantage over ordinary conditional models at the same byte cap.
T: 3 fresh world/init pairs, 4 loads, equal development trials; deterministic full-support retention; no early quality stopping; wall run stops only on time cap. Primary checkpoint4,000 updates; curves secondary.
D: strong useful-Mirror PASS only if at private16 and32, common>=.99 in all3, paired private advantage over EACH primary conditional control >=.02 in all3, with at least one larger all-world95%-quality frontier point. Else FAIL for this tested gate, not impossibility. Time-efficiency PASS requires common>=.99 and >=.02 private advantage against each conditional control in all3 at fixed wall budget. No confidence interval from3 pairs.
C: task-conditioning alone; insufficient transform breadth; shared-weight interference; width fragmentation; optimizer/implementation costs; non-function-preserving insertion.
U: world/init confounded in pairs; limited synthetic task; LR grid not global optimum; FP32/backend/CPU clock jitter; no LLM or general capacity claim.

## Implementation tasks
- [ ] 1. RED/GREEN tests: SM001 operator parity, serialization byte roundtrip, corrupted payload rejection, every model's gradients, private parameter isolation, task ID access, grouped/general forward equivalence, identity FiLM/zero residual parent equality, permutation conjugation cancellation.
- [ ] 2. Training: atomic logs/checkpoints, frozen development selection, data/world hashes, coverage, parent hashes, learning curve times; replay parity tests.
- [ ] 3. Execute development, freeze protocol, isolated calibration, execute new main and controls, actual wall-stopped runs.
- [ ] 4. Reload/audit every saved model, compare predictions, finite data and byte accounting; aggregate per-seed paired differences and rule frontier, immutable evidence ledger.
- [ ] 5. Final report, source, checkpoints, hashes and ZIP; publish important code/results to GitHub when writable, otherwise retain verified local patch and explicitly report no push.

## Review focus
1. Pure permutation conjugation cancels through elementwise GELU: test, not a proposed capacity operator.
2. Do not label task-aware Dense unconditional.
3. Parameter byte cap is not active-compute matching: count both and measure.
4. Common retention is not a proof of single-copy encoding.
5. Width-specific common pretraining must not be falsely called a byte-identical parent across widths.
