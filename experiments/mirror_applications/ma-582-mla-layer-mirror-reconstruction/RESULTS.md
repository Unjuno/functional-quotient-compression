# MA-582 results — MLA latent + layer Mirror reconstruction

## Fact

Two registered WikiText-2 validation worlds used Pythia-70M, four 64-token calibration prefixes, 64-token KV prefixes, and 16 one-token queries each. The frozen configuration used rank-128 PCA and rank-4 residual views with layer groups [0,1,2] and [3,4,5]. A same-seed replay of 58201 reproduced every NLL and serialized byte count exactly. Fresh test seeds 58211–58213 remain sealed.

| Method | NLL delta vs FP16, seeds 58201 / 58202 | Model state | Cache/session | Total, 8 sessions |
|---|---:|---:|---:|---:|
| FP16 KV | 0 / 0 | 0 B | 787,214 B | 6,297,712 B |
| Per-layer MLA rank 128 | +1.041 / +0.375 nat | 1,586,172 B | 99,086 B | 2,378,860 B |
| Group MLA rank 128 | +2.255 / +1.964 nat | 529,418 B | 99,086 B | 1,322,106 B |
| Group MLA + layer View rank 4 | +2.165 / +2.044 nat | 546,844 B | 148,508 B | 1,734,908 B |
| Native group residual rank 4 | +2.165 / +2.044 nat | 546,844 B | 148,508 B | 1,734,908 B |
| Per-layer private residual rank 4 | +2.161 / +1.800 nat | 1,636,098 B | 148,508 B | 2,824,162 B |

The Mirror and native control cache payloads are byte-identical in each seed (both 148,508 B; identical SHA-256). Group sharing cuts 8-session bytes by 44.4% against per-layer MLA and 79.0% against FP16, but NLL misses the frozen +0.05 gate by 1.99–2.20 nat/token. Adding the View lowers group-MLA NLL by 0.089 nat in one seed and worsens it by 0.080 in the other, while adding 66,848 B across model state plus one session (17,426 B model state and 49,422 B cache). The group View is 27.1% smaller than per-layer MLA at eight sessions, at substantially worse NLL.

Group PCA encoder state and residual bases are charged as serialized NPZ files: 529,418 B and 17,426 B. Per-layer PCA state is 1,586,172 B. The full per-seed serialized payloads and hashes are in `ARTIFACT_PROVENANCE.json`; no checkpoints or temporary artifacts were committed.

Compute on CPU: rank-128/rank-4 basis fitting took 1.97/1.98 s; total run wall time 21.58/22.83 s. View encoding was 0.0169/0.0394 s per cache versus FP16 cast at 0.0068/0.0083 s. The rank-4 residual adds a 49,152 multiply-add operation proxy per cached token. Query runtime uses reconstructed cache in the unmodified model and does not represent a fused serving implementation.

## Interpretation

**D: FAIL.** Both dev worlds exceed the NLL gate by a wide margin. Shared layer latents produce meaningful physical-state savings, but lose too much layer-specific information. The residual view is not a stable recovery mechanism and is exactly the native shared-dictionary coefficient representation: this experiment provides no Mirror-specific gain. Group MLA without residual is the smallest state, but the highest-quality configuration among the tested compressed group variants is still far outside the quality gate.

## H / T / D / C / U

**H:** A group-shared rank-128 latent plus layer-specific rank-4 reconstruction View can recover per-layer K/V quality at lower model-state cost.

**T:** Pythia-70M + WikiText-2 valid, seeds 58201/58202; 4 train calibration prefixes; FP16, per-layer MLA, group MLA, group+Mirror, exact native group residual, and private per-layer residual controls. Actual NPZ bytes, cache reconstruction MSE, one-token NLL, fit/encode/query wall time, and operation proxies were recorded. A 58201 same-seed replay matched NLL and byte counts exactly.

**D:** FAIL. Quality gate fails in both worlds; Mirror and native grouped residual arrays are exactly identical. No fresh test run.

**C:** Pooled layer PCA basis rank 128 is too small to preserve layer-specific key/value directions; the residual dictionary has too little rank and is ordinary shared-basis coding. The one-query test may also amplify local cache errors, but does not explain the large reconstruction MSE increase.

**U:** Full-sequence perplexity, longer contexts, other group sizes/ranks, learned projections that replace Pythia K/V weights, and fused serving kernels.

## Fact / Interpretation / Hypothesis

- **Fact:** Group sharing reduces serialized 8-session bytes; it increases validation NLL substantially. The group Mirror and native payloads match exactly.
- **Interpretation:** Storage sharing exists, but this rank/group design loses too much functional information and the residual View adds no distinct mechanism.
- **Hypothesis:** Smaller groups or higher shared rank might improve the storage-quality frontier, but need new registered protocols and controls.
