# SRM003 implementation and experiment plan

Date: 2026-10-07. Status at registration: NOT RUN.

Goal: test learned shared/private FFN residuals without oracle routing or private masks in a small causal Transformer, then physically prune residual capacity and continue learning. Completing this plan means executing and auditing the tests, not forcing the scientific hypothesis to pass.

Architecture: two causal attention blocks, width32, four heads, shared ordinary FFNs. Only the final FFN has a conditional residual bank. Routes depend on contextual hidden states, not external rule IDs. No ES, recurrent loops, rule-address labels, intermediate-state targets or private masks enter training. A private branch is a parameterization label, not proof that it has discovered an independent semantic rule.

## Data and evidence boundary

Synthetic instructions encode an initial 4-bit state and one or two operator tokens, followed by a query. The target is the final state token. Twenty-four operators: eighteen independently sampled bit-permutation/XOR maps and six independently sampled non-affine permutations of sixteen states. The teacher is not generated from the learner's neural basis. Private membership is never supplied to the optimizer, router or pruning selector.

Train/dev/audit operator pairs are disjoint and reverse-closed, defined before learning. All sixteen inputs of every atomic operator occur in training; exact atomic retention is measured separately from unseen pair composition. Main training mixes atomic and two-operation examples equally, using final-output cross-entropy only. This is supervised synthetic endpoint-token prediction, NOT ordinary natural-language pretraining and NOT completely unconstrained rule discovery. A pair-only no-atomic-exposure diagnostic tests the credit-assignment dependency.

Development world66000/init1700 is for implementation and at most two common learning-rate settings. Fresh worlds66001/66002/66003 with initialization seeds1701/1702/1703 are never used for candidate selection. Once development is complete, freeze executable source hashes, widths, schedule and seeds before accessing fresh audit results.

## Comparisons

1. Dense FFN (extra budget spent on final FFN width).
2. Shared FFN plus independent full nonlinear experts with learned top2 routing.
3. Shared FFN plus independent low-rank residual experts with learned top2 routing.
4. Shared FFN plus signed low-rank basis atoms with learned top2 routing.
5. Hybrid: signed shared atoms plus a separate low-rank residual expert bank.

Choose widths/ranks using actual serialized inference bytes only, under the hybrid's measured cap; no outcome-based width search. Record any unspent budget. All models inherit the same shared parent and preserve its function at insertion using zero-output residual initialization. Teacher rule tables are NOT serialized as model weights or needed by the inference decoder.

## Learning and pruning

Shared parent:400 updates. Continue:800 updates to the pruning point, then400 consolidation updates. Batch96, AdamW, cosine decay, clip1. Equivalent token opportunities are controlled; FLOPs and time are recorded separately and are not declared equal.

Hybrid branches after the common pruning checkpoint:
- retain all residual experts;
- greedy leave-one-expert-out validation pruning to half the residual slots, then consolidate;
- random removal of the same number of residual slots, then consolidate;
- same final-sized hybrid trained from the shared parent (small-from-start control).

Pruning uses validation data only and physically slices parameter tensors and router rows. Keep the dense shared path and the shared atom bank. Reset continuation optimizer for all pruning branches, including the no-pruning control. Compare immediate deletion and post-consolidation effects separately. LOO selection work counts as additional training/development work.

## Gates (frozen before fresh evaluation)

- Task readiness: Dense or MoE reaches at least90% atomic accuracy and exceeds25% unseen pair accuracy (chance6.25%) in development. Failure is reported rather than hidden.
- Architecture signal: hybrid improves fresh unseen pair accuracy by at least2 percentage points over byte-capped full-MoE AND low-rank-MoE in every paired fresh setting, with atomic exact-retention loss at most one rule. Otherwise no stable architecture advantage.
- Pruning non-inferiority: selected pruning reduces actual inference bytes, while its unseen-pair accuracy is within1 percentage point and exact atomic retention within one rule of no-prune continuation in every setting. Also report whether it beats random pruning and small-from-start; do not conflate these tests.
- Capacity/LLM superiority: NOT TESTED by this bounded panel. Plateau is not a proof of representational upper capacity.

## Execution checklist

- [ ] Verify recovered SRM001/002 artifacts and repository head.
- [ ] Write failing data, routing, serialization, compaction and causality tests.
- [ ] Implement minimal runnable experiment and pass those tests.
- [ ] Complete development and freeze protocol/source hashes.
- [ ] Run paired fresh comparisons and pruning branches.
- [ ] Run pair-only credit-assignment diagnostic without changing the primary result.
- [ ] Reload all exported models; replay representative full trainings; audit split leakage and file hashes.
- [ ] Publish all executable source, tests, exact counts, limitations and reproducible artifacts; update repository navigation on this branch.

## Accounting and reporting

FP32 CPU execution, no GPU/quantization/fused speed claims. Report hardware/thread count/clock status, optimizer-loop wall and process times, forward examples, backward examples, activated experts, and explicit MAC proxy assumptions. Inference bytes include learned tensors, router rows, tokenizer/config and surviving slot metadata. Raw data/optimizer checkpoints are separate research artifacts.

Report Fact / Interpretation / Hypothesis separately. Three world/init pairs are not confidence intervals. No claim that all arbitrary discrete rules are intrinsically unshareable follows from one narrow model's failure. Preserve all previous records and leave main unchanged.
