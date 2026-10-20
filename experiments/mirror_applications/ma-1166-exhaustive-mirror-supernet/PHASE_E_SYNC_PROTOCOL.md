# MA-1166 Phase E: synchronous parallel Mirror supergraph
**Date:** 2026-10-20 JST. **Status: preregistered before Phase E training.** Original A-D screens unchanged.

**Mirror insertion:** A single 334-coefficient tensor generates virtual 6-8-6 MLP weight vectors for every edge/view with seeded Householder -> tanh -> inverse Householder, permitting 90 distinct logical operations without 90 physically independent 110-parameter MLPs.

## Freeze
- Five virtual state nodes. Every 5x5 directed pair **including self and backward edges** plus five input-to-node anchored links; three Mirror views each: **90 active edge/view candidates**. All edges stay active; **NO pruning**.
- 334 learned physical FP32 scalar coefficients inside ONE `nn.Parameter`: global shared 64, 90 sigmoid edge gates, 180 trainable direction codes. Native virtual weights: 90 x 110 = 9900 logical scalars; **no independent learnable virtual tensors**.
- `mirror_sync`: six source states (x plus five node states) from the PREVIOUS iteration feed all 90 candidate edge MLPs in one batched evaluation; index-add to five target states; repeat three synchronous iterations. Output average of five node states. Contrast `mirror_serial`: same graph, theta and 90 candidate edges per iteration, but sequential target-node (Gauss-Seidel) updates; vectorized sync must also match independent scalar reference (Jacobi). Both have 270 edge-view evaluations per input.
- `affine_sync`: equal 334 learned scalars and graph/compute, direct affine-code virtual-weight decoder (not inverse Householder), same fixed seed mechanism. `native334`: single 6→25→6 residual MLP (331 coefficients) with 3 output group gains, total 334; a strong plain equal-trainable-coefficient baseline but fewer MACs.
- Two synthetic teacher worlds: `mirror_aligned` (separately sampled Mirror sync model with exactly 15 oracle-active edge views; only teacher uses oracle support) and `independent` (15 independent random native virtual MLPs on the same recurrent graph). Student sees ONLY sampled (x,y).
- Independent seeds **509,521**; 2 worlds x 2 seeds x 4 candidates = **16 trainings**. Optional `mirror_sync_t1` fifth model in each pair if compute allows, clearly separate.
- Datasets: train 2048, dev 1024, fresh 1024 independent Gaussian inputs, OOD std1.4 n1024. Adam lr0.015, **320 updates**, batch128, PyTorch CPU FP32 single thread. Every80-update development history; use final checkpoint, never tune with fresh. Loss normalized by mean ||teacher(x)-x||²; no edge penalty. All 90 edge candidates participate in full inference.
- Measure train/dev/fresh/OOD normalized MSE, θ gradient partitions & edge-role interference, exact sync vectorized vs per-edge reference output and gradient, active MAC proxy, actual serialized inference bytes, runtime batch1/batch32, resident generated-buffer bytes, checkpoint hashes/replay.

## Gates
1. Mechanism: output atol1e-5/rtol1e-5, gradient atol2e-5/rtol2e-4 against scalar synchronous reference; all 90 edges x3 steps evaluated; observe speedup over scalar reference at batch32 (sync-vs-serial separate).
2. Mirror-specific: mirror_sync fresh NMSE <=0.95*affine_sync on >=3/4 paired world-seeds AND comparable inference bytes/active MACs, else fail/inconclusive.
3. Sync vs serial: paired fresh NMSE <=1.05*serial on >=3/4 with faster isolated runtime; no guarantee.
4. Native must always be reported. 320-update win is not proof of ultimate capacity, grokking, LM utility, GPU performance or compression; learned representation rank <=334 and resident fixed decoder buffers/program bytes are NOT free.
