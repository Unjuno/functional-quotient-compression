# MA-1166 Phase D — no-prune algorithmic grokking and joint virtual-weight gradient audit

**2026-10-20 JST — preregistered BEFORE Phase D training; previous Phase A/B/C results immutable.**

## Question and falsifiability
Does a fully active virtual 63-edge Mirror supergraph (one physically learned tensor for all virtual MLP weight arrays), compared to equal learned scalar native MLP and same-budget direct affine virtual-weight decoder, show delayed algorithmic generalization (*grokking*) more frequently or earlier?

**Do not confuse** smooth improvement in test CE with grokking. Grokking must exhibit **training memorization BEFORE held-out generalization** followed by sustained sharp recovery after >=3x the memorization step. Record train/test accuracy and cross-entropy jointly.

## Frozen task
- Modular addition (a+b) mod 17. All 289 ordered pairs; fixed pseudorandom 45% train fraction (130 pairs), 55% held out. Input operands share one learned 17×3 embedding; concatenate 2×3 => 6 features. Shared 6→17 classifier including bias.
- Body: original Phase C `VirtualSupernet('mirror')`, all 6 unrolled node events, all 21 forward node-event pairs (all non-adjacent skip links included) ×3 view addresses = 63, each a 6→8→6 native virtual MLP, **one** trained 253-float `theta`. Gate probabilities remain continuous, no sparsity regularization, no hard pruning at train OR inference. Each virtual 110-vector generated from the shared 64-coefficient core via two learned-code Householder nonlinear-inverse Mirror transformations.
- Baselines: direct affine virtual-weight decoder with same learned core/codes/edges and exactly 253 body coefficients; native `NativeMLP` 6→19→6 with exactly 253 body coefficients. Same learned operand-embedding (51 coefficients) and classifier (119 coefficients) attached to every body; hence 423 learned scalars **for all methods**. Independent initialization of native body; embedding/head initialization identical for paired seeds.
- Paired data/optimizer seeds **421, 433**; all examples used in full-batch gradient, identical train/test data. `AdamW(lr=0.008, weight_decay=0.5)`, cross-entropy loss; exactly 6,000 updates; evaluate at step0, every100 updates and last (full train and heldout).
- CPU PyTorch FP32 single-thread. Save complete metrics trajectories, training timing, post-training checkpoints plus sha. Never use test performance for hyperparameter tuning.
- Grokking-event criterion: first step train accuracy ≥98%, held-out accuracy at that point ≤35%, subsequent (>=3× later) held-out accuracy ≥90% for >=3 successive checkpoints. If no model satisfies by 6,000, classify NOT OBSERVED WITHIN BUDGET, not impossible.
- Separately verify mathematical parallel training: all 63 virtual outputs participate in one reverse-mode call; log nonzero grad norms for global 64 core, 63 gates and 126 codes, and check \(\nabla_\theta \sum_e L_e =\sum_e\nabla_\theta L_e\) numerically on a linear sum of per-view readout probes. This does not mean sequential Loop events are parallel in execution.
- Physical vs virtual degrees: 110×63=6,930 virtual weights; physical 253+170=423 learned coefficients. Decoders and seeded directions are part of inference program, not independent learned parameters. No claim of arbitrary independent virtual weight capacity.
- Resource bound: checkpoint and curve available for every completed run; if runtime prohibits full budget, clearly mark incomplete and don't assert grokking absence.
