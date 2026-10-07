# MA-199 — task-specific coordinates in a shared gradient plane

Status: SCREENING  
Branch: `research/ma-199-gradient-coordinate-20261007`  
Base commit: `a5dc84c59a6a7fd35c43dc40003f3be52cc03d75`

## H — falsifiable hypothesis

When several task updates share a two-dimensional left gradient plane, one per-task Givens angle and one 8D update vector will preserve aligned task performance with less serialized inference and optimizer state than independent rank-1 LoRA. Unrelated gradient directions should require private capacity.

## Prior-art delta

FedLore (arXiv:2610.01620) shares an evolving projected-gradient basis within each federated round, keeps Adam moments in projected coordinates, and refreshes the basis across rounds. Its paper reports two moment tensors of size 2rn and emphasizes that sharing client projections removes projection/heterogeneity coupling. GaLore (arXiv:2403.03507) likewise compresses optimizer state by projecting gradients. OGD (arXiv:1910.07104) and iGSP (arXiv:2605.19301) protect prior skills by projecting gradients against stored task subspaces. PA05/PA06 establish generated low-rank adapters as alternatives.

This screen does not reimplement federated training or claim that a random plane can replace data-driven basis discovery. It tests the narrower parameterization: after paying for a shared plane, does a task-specific gradient-view angle plus a vector encode aligned per-task updates more cheaply than LoRA, while preserving old skills? The generic rank-2 coefficients in the same plane are the key simple shared-basis control.

## T — protocol

One frozen 16x8 base and a charged, seeded 16x2 orthonormal plane support four sequential skills. Aligned task deltas are rank one with left directions inside that plane. Independent task deltas are unrelated dense matrices. Old task states freeze as new tasks arrive.

The candidate reconstructs each delta as `(P @ [cos(theta), sin(theta)]) @ v_t.T`. Controls are hard tying, ordinary `P @ C_t` coefficients, independent rank-1/2 LoRA, and independent full weights. Development uses seeds 19901/19902, 64 examples per task, 300 updates, and LR .003/.01. Fresh seeds 19911–19913 remain sealed until the registered development gate passes.

Every payload charges the base, plane, codes, task IDs, metadata, and optimizer state where applicable. Quality, forgetting, actual bytes, update examples, MAC proxy, wall time, inference throughput, and basis setup cost are recorded.

## Gates

Fresh PASS requires all three aligned worlds to reach <=1.10x rank-1 LoRA final MSE, <=0.90x its total inference bytes, <=0.75x its incremental resume-state bytes per skill, and <=0.01 mean prior-task MSE increase. If ordinary shared-plane coefficients match at no more bytes, Mirror-specific benefit is not established.

This is fixed-update synthetic evidence. The supplied basis discovery cost and natural-task generality are outside scope.

## Results

Pending protocol freeze and development.
