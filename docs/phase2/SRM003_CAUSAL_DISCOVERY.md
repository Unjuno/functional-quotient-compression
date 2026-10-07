# SRM003 — Causal shared/private discovery and pruning

Date: 2026-10-07 JST. Evidence: two-layer synthetic causal Transformer, width32, four heads, final-answer-token cross-entropy. NOT natural-language pretraining, not a capacity multiplier or equal-FLOPs result.

## Execution versus scientific outcome

The registered implementation/comparison/pruning/long-control/replay plan was executed. The scientific architecture-adoption and task-readiness gates FAILED. Do not convert completed execution into a successful hypothesis.

Original preregistration commit: `9f03af294dc43ca3c53fef23a429f321813d9438`. Development world66000/init1700 tested common learning rates0.003/0.001. Both failed the25% unseen-pair readiness criterion;0.003 was better for all five development methods. Before fresh evaluation, freeze source hashes and add4800-update long controls and a predicted-intermediate replay diagnostic. Primary runs were not changed afterward.

Fresh worlds66001/66002/66003, initial seeds1701/1702/1703. Parents400 updates, specialization800, consolidation400, batch96, AdamW cosine decay. Teacher:24 permutations of16 states,18 sampled bit-permutation/XOR rules and6 random rules outside that family (the realized6 were audited as non-affine). No oracle router, private mask, intermediate-state target, ES or recurrent training loop. Operator tokens remain explicit inputs: this is not unstructured language discovery.

Unordered rule pairs are hash-split into train/dev/audit; reverse orders stay in the same split. All16 states are enumerated for each held-out pair. Single-rule examples are mixed50/50 with two-rule examples, and every single-rule input is exposed in training. Atomic retention is an exhaustive24x16 test, not accuracy of compound examples merely containing a rule.

## Main fixed-update results

Median [min,max] over3 paired worlds/init settings; not confidence intervals. All five main models retain24/24 atomic rules exactly.

|Method|Actual inference bytes|Unseen pair accuracy at1600 updates (%)|
|---|---:|---:|
|Dense (final FFN width107)|90,217|10.05 [10.00,16.03]|
|Shared base +4 full FFN experts, top2|90,111|9.93 [9.81,15.11]|
|Shared base +8 rank6 residual experts, top2|88,745|10.50 [9.80,15.98]|
|Shared base +8 signed rank6 atoms, top2|88,742|10.50 [8.76,14.45]|
|Hybrid:8 rank2 signed atoms +8 rank4 residual experts, top2 each|90,329|10.56 [9.56,15.57]|

Hybrid minus full-MoE paired differences:+0.750,+0.461,-0.368 percentage points. Hybrid minus low-rank-MoE:+0.063,-0.410,-0.245 points. The preregistered2-point improvement in every fresh setting did not occur.

Configurations were selected by actual file lengths only under90,329 B, not by audit quality. Inference payloads exclude teacher tables, world seeds and optimizer state; include all learned tensors, routers, slot indices and required configuration. Low-rank comparators also update the base and are not frozen-base LoRA reproductions.

## Long controls and the decisive diagnostic

At6400 cumulative updates:

|Method|Unseen pair accuracy median [min,max] (%)|Exact atomic rules|
|---|---:|---:|
|Dense|14.50 [14.46,18.75]|24/24|
|Full-MoE|14.71 [13.19,16.09]|24/24|
|Hybrid|15.99 [14.12,17.78]|24/24|

Hybrid exceeds full-MoE in3/3 by0.94–1.69 points, but beats Dense in only1/3. Absolute quality remains low. Post-hoc seen-pair accuracy is also incomplete:36.24–55.66% across these models. This is not an upper bound on representational capacity or a pure unseen-generalization failure.

Use the same model to predict the output of the first atomic instruction; feed that predicted state token and the second instruction into the same model. No true intermediate state is supplied. All9 long-control models score100% on the unseen pairs in this **two-forward external-composition counterfactual**; the same9 models at1600 updates also score100%.

Interpretation: the needed atomic mappings are present, but direct end-to-end composition is not acquired in this tested architecture/training regime. The counterfactual has extra compute and supplied execution order. It is not a one-forward result, not a new capacity claim, and not recurrent training.

This clarifies SRM002: the recovered `SharedBasis.forward` explicitly iterated over the supplied operator list and passed the updated state to the next operator. SRM002 demonstrated learning operators WITH an executor. SRM003 asks a conventional causal decoder to discover its own execution structure. Parallel weighted residual summation is not automatically ordered function composition.

## Pruning

From the1200-update Hybrid, remove4 of8 additional residual slots using26 greedy leave-one-out development-pair CE evaluations. Preserve the shared path and signed atom bank. Reset optimizer for400-step consolidation in selected, random, and no-prune branches.

|Method|Bytes|Median unseen pair accuracy (%)|
|---|---:|---:|
|Keep all|90,329|10.56|
|Validation-selected prune +consolidation|85,672|10.63|
|Random prune +consolidation|85,672|10.29|
|Same final size from shared parent|85,672|10.37|

All retain24 atomic rules. Selected minus no-prune paired changes:+0.0625,-0.9221,-0.5515 points: within the numerical1-point noninferiority tolerance. However selected pruning beats random only1 win/1 tie/1 loss, and small-from-start only1/3. This is not useful-quality compression or proof that pruning discovers semantic private rules. The physical saving is4,657 B (5.16%), not half the whole model. Active top2 per bank remains unchanged.

Pair-only training without atomic examples gives median unseen-pair9.19% for Dense and8.95% for Hybrid, with0/24 atomic rules exactly retained in all3 settings. This intervention also changes input-length/query-position exposure; it does not isolate a single cause by itself.

## Compute and audit

CPU AMD EPYC9V74, PyTorch2.10.0+cpu, FP32, one thread per worker. Main runs used concurrent workers; no wall-time learning superiority is claimed. Separate isolated calibration: batch96, five token slots,10 warmups then40 updates/group for5 groups, clock not locked (snapshot about2596 MHz). Median update times: Dense5.360 ms [4.976,6.128], full-MoE9.805 [9.123,10.777], Hybrid10.523 [9.836,11.563], pruned11.690 [10.050,12.081]. Smaller MAC proxies did not yield faster execution; pruning speedup was not demonstrated. These are microbenchmarks, not equal-time training.

19 new experiment tests pass.182 inference payloads roundtrip byte-exact;52 final-model metric recomputations differ by0. Original frozen training-source hashes unchanged. Dense and Hybrid replay their full1600-update path byte-exact; selected pruning and400-step consolidation also replay byte-exact. Train/dev/audit pair sets are disjoint and reverse-closed. Same-session audit, not external replication; historical repository-wide tests were not rerun because historical source was not modified.

## Decision / next question

Retain the shared/private idea as a hypothesis, not an adopted solution for LM-only composition. Do not scale atom counts or claim natural-language compression from this result. The next bounded question is whether a fixed-depth, explicitly checked intermediate-state interface can connect learned rules without oracle intermediate targets or recurrent loops. Existing controlled SRM001/002 results remain preserved, but their executor and task scaffolds must be stated.

[Runnable experiment](../../experiments/shared_rule_moe/srm003_20261007/README.md). [Raw fresh counts](../../experiments/shared_rule_moe/srm003_20261007/ALL_RESULTS.csv). Full checkpoints, data, Japanese report and audit logs are in the conversation artifact SRM003_CAUSAL_DISCOVERY_RESULTS_2026-10-07.zip.
