# SM002 — conditional-expert controls, equal-time comparison and late-LR diagnosis

Date: 2026-10-07 JST. Evidence lane: SYNTHETIC / actual serialized inference payload. Status: mixed evidence; the preregistered strong Mirror-superiority gates were NOT met.

## Scope and contract

SM002 follows the recovered SM001 finite-rule MLP, not the 12-layer Transformer lane. All models receive task ID, rule ID and input value; no learned router and no answer-derived selection. There are four tasks, four shared random-permutation rules, and 4/8/16/32 independent private rules per task. Each rule maps 32 values to 32 values. Evaluation exhaustively audits learned finite-domain retention, not unobserved random-label extrapolation. Common/private examples are balanced 50/50.

The cap is 44,155 actual serialized bytes, including FP32 learned tensors, config, shape metadata, method, transform seed and strength. No padding manufactures equality. Different architectures use the largest width up to64 that fits. Generic decoder/library code is externally fixed; optimizer state is not inference storage. Same-width architectures copy identical common-only parents; different-width parents are necessarily different. Each parent receives1,000 updates at lr0.003; all24 parents reached100% common accuracy. Fixed shear/gate insertion is not function-preserving; initial quality is logged.

Development uses world92001/init171 and exactly three learning rates per primary method (.001/.003/.01). All eight select .01. Configs and the actual-time protocol are frozen before audit worlds22001/22002/22003 with initial seeds201/202/203. Main continuation is4,000 AdamW updates, batch128, weight decay.01, gradient clip1. All methods use identical master-stream prefixes. Main development/search outcomes were not selected on audit worlds.

Counts:24 development +111 main +24 actual-time runs =159. An explicitly outcome-selected diagnostic adds three constant-LR replays and three late-LR interventions, giving165 runs. The24 parent trainings are separate.

## Fact: private32 at the same4,000 updates

Private accuracy is the median and full range across three paired world/init settings; these ranges are NOT confidence intervals.

|Method|Width|Bytes|Private accuracy %|
|---|---:|---:|---:|
|Task-aware Dense|64|44,155|69.43 [68.73,70.09]|
|Assigned shear Mirror|64|44,155|72.63 [72.44,73.49]|
|Fixed direct post-GELU gate|64|44,155|68.55 [67.48,68.63]|
|Learned FiLM|59|43,470|60.55 [59.20,60.74]|
|Task rank-2 residuals|55|43,867|67.68 [66.38,68.38]|
|Shared trunk/private heads|44|43,933|60.38 [59.16,61.30]|
|Shared input/output, private middle FFN|40|43,680|64.23 [63.28,66.94]|
|Independent narrow experts|21|43,703|54.17 [53.66,58.15]|

Mirror exceeds each primary conditional comparator in all three private32 pairs. Against fixed gate, its paired advantage is +4.86 pp median [+3.88,+5.15]. However, common accuracy in one Mirror run is506/512=98.828125%, one correct input below the507 needed for the99% gate. Do not portray that shortfall as a large common-rule collapse.

Input-partition shear, which assigns by rule/input rather than task, obtains73.46% median [73.14,74.29], and assigned Mirror wins only1/3. Thus semantic task assignment has no demonstrated special advantage. Large same-width independent experts obtain99.95% median but require175,114 B and are outside the cap.

At private32, assigned Mirror retains only [1,0,0] of128 private rules at the rule-level95%-accuracy threshold (31 of32 values). High mean input accuracy is not128 fully retained rules.

## Fact: actual3.5-second optimizer-loop budget

This budget was fixed from development-only calibration before main evaluation, not selected using audit outcomes and not inferred by converting old timing ratios to updates. Runs stop after the first complete update reaching3.5 seconds; maximum observed overshoot was below0.962 ms. This measures optimizer-loop elapsed time, including forward/backward, clipping, indexing and optimizer work; it excludes pretraining, data generation, evaluation and serialization. It is neither end-to-end time nor equal FLOPs.

Actual hardware: virtual AMD EPYC9V74, PyTorch2.10.0+cpu, FP32 eager, batch128, one thread and one sequential worker. Clock was not fixed. Three concurrent main workers had finished before these measurements. Calibration used30 warmup updates and7 groups of100 updates; median update times were Dense0.760 ms, Mirror1.154 ms, gate0.805 ms.

|Method|Private accuracy %, median [min,max]|
|---|---:|
|Dense|70.00 [69.36,70.97]|
|Mirror|69.38 [67.55,69.80]|
|Fixed gate|68.29 [68.16,70.83]|
|FiLM|57.32 [56.10,58.67]|
|Rank-2 residuals|60.77 [60.55,61.35]|
|Private heads|62.52 [61.25,64.60]|
|Private middle FFN|65.11 [63.87,66.24]|
|Independent narrow|54.96 [52.61,58.94]|

The paired Mirror-minus-gate differences are +1.514,-1.440,-0.610 pp, so the paired median is-0.610 pp and Mirror wins1/3. The difference of separate method medians is not the median paired difference. Mirror also wins only1/3 against Dense, while still exceeding the other tested conditional architectures in3/3. This is a failure to demonstrate stable superiority over a cheap gate, NOT proof of equivalence or universal inferiority to all MoE models.

## Fact: final-checkpoint failure can be late optimization degradation

At private16, the final4,000-update Mirror accuracies are100%,84.86%,100%; private middle FFN gives100%,99.90%,100%. But the failing Mirror run had already achieved100% common/private accuracy at2,000 through3,500 updates. Its final decline cannot be interpreted as inability to store those rules.

A separate post-hoc diagnostic selected that Mirror case and corresponding late-degrading gate/Dense cases. Replaying the unchanged4,000-update protocol reproduces the original snapshots byte-for-byte. Changing only lr from.01 to.001 after update3,500, without resetting AdamW, gives:

|Selected case|Constant LR final private %|Late-LR reduction final private %|
|---|---:|---:|
|Mirror, world22002|84.86|100.00|
|Gate, world22001|76.32|100.00|
|Dense, world22001|87.26|99.90|

All intervention common accuracies are100%. This supports a late-LR effect in these three selected cases; it is not new-world confirmation or a complete Adam/decay/interference mechanism. Primary results and preregistered gates are unchanged. Note also that demanding a2 pp advantage at a load where a comparator is already near100% encounters an accuracy ceiling; failure of that strong gate is not a capacity-impossibility result.

## D/C/U and evidence boundary

The strong useful-Mirror gate and time-efficiency gate FAIL for the tested protocol. A general storage-capacity claim remains UNCERTAIN. Supported: high-load fixed-update private-accuracy advantage over the tested conditional controls; no stable advantage over cheap gate at actual equal loop time; no task-assignment-specific gain; selected-case late-LR stabilization.

Limitations: only three world/init pairs (world and initialization not independently crossed), finite random-lookup task, limited hyperparameter search, unequal active compute and width-specific parents, non-function-preserving transform insertion, FP32/CPU dispatch costs, unfixed clocks, outcome-selected diagnostic. No natural-language, Transformer, token-period or real-world capacity conclusion follows.

Verification:35 tests;165 final models and1,322 curve snapshots reloaded;24 parents;169 main/time prefix snapshots byte-exact;45 diagnostic equality checks; three full4,000-update replays. This is in-session self-audit, not external replication. All frozen source and plan hashes were checked. Historical MN010/MN011 and the unverified39/48 long-sequence record are not altered or used.

Core source, immutable plan, exact per-world counts and audit summary: [experiment directory](../../experiments/specialized_mirror/sm002_20261007/). The full conversation ZIP contains every checkpoint and audit script.

Relevant primary literature: FiLM arXiv:1709.07871; LoRA arXiv:2106.09685; sparsely gated MoE arXiv:1701.06538. The rank-2 control co-trains the base and is not original frozen-base LoRA; deterministic task addresses are not a learned sparse router.
