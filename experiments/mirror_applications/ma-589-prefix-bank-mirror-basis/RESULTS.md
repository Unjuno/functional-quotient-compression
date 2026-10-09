# MA-589 results — shared basis for a KV prefix bank

## Fact

Two registered WikiText-2 validation seeds each evaluated 16 independent 64-token text-context prefixes using Pythia-70M. Thirty-two train contexts per seed fit a shared rank-128 per-layer latent basis and rank-4 residual dictionaries. A same-seed replay of 58901 reproduced all method NLLs and cache byte counts exactly. Fresh test examples were not loaded.

| Method | NLL delta vs FP16, seeds 58901 / 58902 | Model state | Cache/context | Total, 16 contexts |
|---|---:|---:|---:|---:|
| FP16 KV | 0 / 0 nat | 0 B | 787,214 B | 12,595,424 B |
| Shared MLA rank 128 | +0.260 / +0.389 nat | 1,586,172 B | 99,086 B | 3,171,548 B |
| Mirror residual rank 4 | +0.150 / +0.334 nat | 1,636,098 B | 148,508 B | 4,012,226 B |
| Native shared residual rank 4 | +0.150 / +0.334 nat | 1,636,098 B | 148,508 B | 4,012,226 B |

The Mirror bank uses 31.9% of FP16 bytes at 16 contexts (68.1% less storage), but misses the +0.05 nat/token quality limit in both seeds. Its NLL is 0.110/0.055 nat better than shared MLA rank128, yet its serialized cache archive is byte-identical to the native shared residual representation in both seeds. Actual basis, residual, cache, and metric file sizes/hashes are recorded in `ARTIFACT_PROVENANCE.json`.

The rank-4 method charges 1,636,098 B model state and 148,508 B per context; the rank-128-only method charges 1,586,172 B and 99,086 B/context. The serialized basis and code payloads are included in these totals.

Basis fitting took 6.73/6.53 s; total wall time was 24.51/25.96 s. Mirror cache encoding took 0.0153/0.0218 s versus FP16 conversion 0.0062/0.0046 s. Measured query calls took 0.0630/0.0675 s with reconstructed caches versus 0.0155/0.0144 s with FP16 caches. These are CPU PyTorch timings without a fused serving kernel. The Mirror reconstruction adds a 49,152 multiply-add proxy per cached token.

## Interpretation

**D: FAIL.** The byte frontier improves substantially when amortized over multiple contexts, and the rank-4 residual improves NLL over rank-128-only compression in both seeds. However, it still fails the preregistered NLL tolerance and exactly aliases a native shared-dictionary control. Thus there is no Mirror-specific result, and the storage reduction trades away unacceptable next-token quality under this gate.

## H / T / D / C / U

**H:** One shared basis plus compact per-context residual views can preserve context-prefix quality while reducing aggregate prefix-bank bytes.

**T:** Pythia-70M and WikiText-2; 32 train cache prefixes fitted shared per-layer rank-128 PCA and rank-4 residual dictionaries; two registered validation seeds, 16 query contexts each; FP16, shared latent, Mirror residual, and exact native residual controls. Actual serialized payload, NLL, reconstruction error, fit/encode/query time and operation proxies were recorded. Same-seed replay matched core metrics exactly.

**D:** FAIL. NLL deltas are +0.150/+0.334 nat/token versus a +0.05 gate, and Mirror/native cache payloads are byte-identical. Fresh stayed sealed.

**C:** The remaining error is caused by rank-limited post-hoc cache compression; the residual rank-4 dictionary restores some quality but not enough. The apparent gain over shared MLA is ordinary shared-basis residual coding.

**U:** Learned continuous Prefix-Tuning vectors, multi-token continuation perplexity, longer prefixes, task-conditioned adaptation, other ranks, and optimized serving kernels.

## Fact / Interpretation / Hypothesis

- **Fact:** 16-context state drops from 12,595,424 B to 4,012,226 B; NLL misses the frozen gate; native and Mirror archives are exact matches.
- **Interpretation:** There is a cache storage-quality tradeoff, not Mirror-specific prefix-function multiplicity.
- **Hypothesis:** Task-weighted or trained prefix representations may shift the frontier, but require a new MA protocol and native controls.
