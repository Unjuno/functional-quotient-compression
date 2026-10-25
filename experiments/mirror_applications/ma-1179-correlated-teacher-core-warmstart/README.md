# MA-1179 — Independently pretrained correlated teachers: pooled-core Mirror distillation

**Mirror insertion:** Add a learned rule address `m` at the hidden feature representation of one shared 4→16→4 tanh MLP, creating three functional rule views without three full physical copies.

## Pre-training research hypothesis (H)
A shared Mirror core found **from function outputs alone** can distill *independently initialized* teachers of related tasks. A 250-update pooled-core functional warm-start followed by 500 conditional updates may improve Mirror's optimization versus 750-update random conditional training. **Mirror-specific value** requires outperforming equally sized FiLM and rank-one functional codes, not simply benefiting from pretraining.

## Controls and environments (T)
- Three worlds: exact Mirror-constructed positive control; independently pretrained teachers with 75% common/25% specific analytic target; unrelated independently pretrained teachers.
- 5 seeds; main hidden size 16; 8 methods = 120 student checkpoint models.
- Mirror/FiLM/rank1: 196 floats each, independent 201 floats, shared 148 floats. Warm/cold steps both total 750 × batch128; pooled warm stage uses identical training labels and adds no privileged teacher weights.
- CPU FP32, one PyTorch thread. Train/dev → SHA256 seal → unseen fresh/OOD. Compare Fresh per-rule normalized MSE, training time including pooled cost, inference time and actual serialised inference bytes.

## Frozen primary decisions (D)
- Warmstart gate: Mirror warm beats Mirror cold by ≥10% Fresh NMSE in at least 4/5 correlated-world paired seeds.
- Mirror-specific gate: Mirror warm beats **both** warmed FiLM and warmed rank1 by ≥10% Fresh NMSE in at least 4/5 correlated-world paired seeds.
- Deployment gate: no claim unless actual quality+bytes+CPU runtime Pareto evidence supports it. Exact-Mirror world's wins do not establish general advantage.

## Contradictions (C) / uncertainty (U)
A simpler FiLM/low-rank control may represent the correlated task family as well or better. Pooled pretraining may impair specialization; 5 seeds, only one 4D synthetic task family, and CPU-only timing leave substantial uncertainty. Report FAIL unchanged if either primary gate misses.

## Closest prior work
MA-1178 pretrained-MLP Mirror distillation (fixed Mirror/FiLM/independent teacher worlds and posthoc teacher-weight transfer); MA-1176 5-axis scaling. This experiment differs by **independently pretrained correlated teachers**, no teacher-weight transfer, and a compute-charged pooled-function-only warm start with matched low-rank code control.

Protocol: `PROTOCOL.json`. Source: `source/run.py`, tests `tests/test_run.py`. All counts and decisions fixed before launching training.
