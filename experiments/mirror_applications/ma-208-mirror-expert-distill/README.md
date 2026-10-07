# MA-208 — distill specialist teachers into a Mirror view bank

Status: SCREENING  
Branch: `research/ma-208-mirror-expert-distill-20261007`  
Base commit: `621dec2ffdf55c3f0503dd75695ea27fdc68544c`

## H — falsifiable hypothesis

A shared student MLP with one learned hidden-pair Givens coordinate per specialist can distill related teacher experts at ordinary multi-head KD quality with substantially fewer inference bytes. Independently sampled teacher functions should show where private student weights are needed.

## Prior-art delta

MoE-to-dense work such as *Pruning and Distilling Mixture-of-Experts into Dense Language Models* (arXiv:2605.28207) selects expert knowledge, initializes a dense student, and distills teacher logits. Recent multi-teacher on-policy distillation work—including *Rethinking Self-Distillation for Multi-Teacher Capability Merging* (arXiv:2610.04272) and verifier-gated teacher assignment (arXiv:2609.15404)—consolidates specialist capabilities in a single student. PA09 uses a shared trunk with multiple future-token heads; PA10 conditions parallel token prediction on sequence randomness. Those works establish shared-trunk, multi-head, and teacher-selection controls, but do not test whether expert functions related by a small hidden Mirror coordinate can replace private student heads.

## T — protocol

Five task IDs use 12D inputs, 16 hidden units and four output classes. Task 0 defines a shared MLP. In the aligned family, later teachers use the same MLP and apply a Givens rotation to the first hidden pair before the shared output projection. In the independent family, every teacher MLP is sampled independently.

Distill each teacher's temperature-2 softmax distribution using forward KL on 128 task examples for 300 updates. Controls: frozen one-student hard tie; sequential single-head KD; ordinary per-task output-head KD; rank-2 LoRA output updates; Mirror hidden views; independent full student per task; and teacher functions as an upper quality/bytes reference. Development seeds 20801/20802 choose LR .003/.01. Fresh seeds 20811–20813 stay sealed until the registered development gate passes.

The inference codec charges all shared and private tensors, task IDs, method names, dtypes, shapes, and headers. Resume optimizer bytes are reported separately. Quality includes held-out teacher KL, top-1 agreement, ECE against teacher labels, and retention by task.

## Gates

Fresh PASS requires all three aligned worlds to meet: teacher KL <=1.10x multi-head and <=1e-4 absolute, top-1 agreement >=0.99, ECE within 0.02 of multi-head, and total serialized inference bytes <=0.50x multi-head. Rank-2 LoRA is the byte-near control. A match by a simpler control precludes a Mirror-specific claim.

This synthetic fixed-update distillation screen does not establish language-model expertise, independent capacity, or a near-converged storage frontier.

## Results

Pending protocol freeze and development.
