# MA-794 — Factorized depth × expert product views

## Fact

The frozen seed-79401 synthetic screen used eight layers, eight global experts, 8×8 linear operators, rank-2 shared matrix bases, and checkerboard heldout layer/expert roles. The product-code and native bilinear controls serialize byte-identical NPZ payloads (3,966 B, SHA-256 `df3bf048652269ce28e0673a016167ccac00291b37be30c981384e01d0672e8c`) and produce identical heldout operator nRMSE .05207 and all-role output nRMSE .04176. Independent roles use 15,925 B with zero oracle reconstruction error; hard global tying uses 2,690 B and heldout nRMSE .07912; additive depth/expert codes use 3,959 B and heldout nRMSE .08105. Every fixed per-layer router assigns one example per expert (min=max=1). Optimizer updates: zero. Compute is a synthetic dense MAC proxy; measured CPU operator evaluation is tens of microseconds for only 512 basis-vector examples and is not a throughput claim.

## Interpretation

Product coordinates reconstruct a compact shared-basis family from observed roles, but do not meet the registered .02 heldout error gate. They save 75.1% bytes versus independent full role matrices, while the native bilinear baseline is exactly the same method and payload. The additive control has marginally lower bytes and a weaker heldout score, so no strict frontier advantage is established. Balanced routing is imposed by a fixed identity router and does not validate learned router balance.

## H / T / D / C / U

- **H:** Rank-2 depth×expert product coordinates over a shared expert pool recover heldout layer/expert functions with ≤.02 nRMSE and at least 25% less complete storage than independent roles, while preserving balanced routing.
- **T:** Seed 79401; 8 layers, 8 global experts, dimension 8; rank-2 operator basis; checkerboard role holdout. Compared independent role matrices, hard expert tying, additive depth/expert codes, Mirror product codes, and native bilinear shared basis. Measured actual compressed NPZ payloads, heldout/all-role errors, balanced fixed-router loads, MAC proxy, reconstruction MAC proxy, 512 basis-vector examples, CPU evaluation wall time, zero updates.
- **D:** **FAIL** for Mirror-specific value. Heldout error misses the .02 gate; Mirror exactly aliases native bilinear; it does not strictly improve the additive control's bytes/quality frontier.
- **C:** This is ordinary low-rank bilinear factorization under a rank-2 synthetic teacher. The poor heldout fit may arise from the small single-seed alternating fit and basis/intercept estimation, while the payload advantage comes from low-rank model structure rather than a Mirror-specific mechanism.
- **U:** Fresh role compositions, trained MoE or MoRE, language-model loss, learned routing and load balancing, optimizer/update effects, broader seeds, private residual crossover, GPU latency, and near-convergence capacity remain untested. Fresh access stayed sealed after exact native alias and failure of the development gate.

## Scope

This is a synthetic linear mechanism/storage screen, not a trained neural-network or LM result. The independent condition stores oracle teacher roles; fitted methods use only observed role matrices. Routing uses a fixed identity map. See [the frozen protocol](PROTOCOL.json) and [core results](RESULTS_CORE.csv).
