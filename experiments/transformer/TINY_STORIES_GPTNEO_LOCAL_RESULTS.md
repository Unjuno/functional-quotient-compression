# TinyStories GPT-Neo scale-series: local actual-checkpoint results

This note records high-value results obtained in the local container from four user-supplied GPT-Neo checkpoints (`1M.zip`, `3M.zip`, `8M.zip`, `28M.zip`). The ZIP archives themselves are **not** stored in the repository.

These are real checkpoint-weight experiments, but the functional replay used a local PyTorch implementation of GPT-Neo plus a local GPT-2 BPE implementation because `transformers` was not available in the container. The tokenizer was spot-checked against standard GPT-2 token IDs (for example `Hello -> 15496` and `Once upon a time -> [7454,2402,257,640]`). The local forward has not yet been cross-checked against an independent Transformers runtime, so the results below are evidence for research direction rather than a final codec certificate.

## 1. Provenance

| label | ZIP SHA-256 | `pytorch_model.bin` SHA-256 | hidden | layers | heads |
|---|---|---|---:|---:|---:|
| 1M | `6e95174c90a9a2a9a8e5f97d934e78e2863d6b88e9450c7b9a0ff4c25e0e1b16` | `07f9609ea882b8163ff3b23d40e2b82cb715d409631beb15c84b164f3877dae7` | 64 | 8 | 16 |
| 3M | `a6a7328ab03fa56a4e13de55ae8a6b9284f8166cdcb1c37574f82d0d685760ce` | `0e4f93a86407a3f685520e48481a236c7b5e3a4f8e4fc8d1e58f3a748d97a37c` | 128 | 8 | 16 |
| 8M | `d1a702e539bbb44393b092bbcbe3af9d3e16f9533284b55c1506a6196834c180` | `22c355bfabebc1f6c861b3f5d7a801e96c7f6da4af4bb0f7780096ab82ea6716` | 256 | 8 | 16 |
| 28M | `a031f4af41f515789cf9d9e3f52bdd1540a711e5a6486072c0a6e5e32b1a5474` | `8ddd260f51b439744c8cc785b5516327d4bf32e31ccbfa9009bfadf12557fcf5` | 512 | 8 | 16 |

All use GPT-Neo alternating global/local attention, window 256, vocabulary 50,257. Floating-state scalar counts (excluding non-floating causal-mask buffers) are 3,745,992 / 8,278,408 / 19,702,536 / 51,987,976. The token embedding accounts for approximately 85.9% / 77.7% / 65.3% / 49.5% respectively.

## 2. Uniform ~0.23 bit/scalar embedding PQ fails functionally

A matched product-quantization family (32-dimensional blocks, K=128, 4-bit codebooks, packed indices) was applied to the tied input embedding / LM head. On the same fixed local probe:

| model | embedding bps | embedding NMSE | mean KL | top-1 flip |
|---|---:|---:|---:|---:|
| 1M | 0.2391 | 0.3887 | 5.717 | 1.000 |
| 3M | 0.2340 | 0.5044 | 4.252 | 0.979 |
| 8M | 0.2315 | 0.5859 | 4.690 | 1.000 |
| 28M | 0.2302 | 0.6591 | 4.463 | 1.000 |

Thus scaling alone does not rescue uniform weight-space embedding PQ at the 64x-like rate.

## 3. Functional root + sparse private exceptions is much better

For the 8M checkpoint, token rows were first grouped by their **calibration logit signatures** (`H_cal E^T`) rather than Euclidean weight proximity. A shared root codebook was then combined with sparse per-token quantized residual exceptions chosen by an output-Fisher proxy.

At a fixed ~0.25 bit/scalar embedding budget, root complexity and exception precision have a discrete joint optimum. Representative single-probe results:

- K=256 root + 4-bit exceptions (2,385 tokens): KL `0.945`;
- K=512 root + 4-bit exceptions (2,090 tokens): KL `0.914`;
- K=1024 root + 4-bit exceptions (1,545 tokens): KL `1.027`;
- K=384 neighborhood: KL approximately `0.89–0.99` depending on root initialization;
- at K=384, uniform 4-bit exceptions outperform 2/3/5/6/8-bit uniform alternatives;
- a small mixed-precision refinement (top ~384 Fisher-ranked exceptions at 5 bit, remaining exceptions at 4 bit) gave KL about `0.910` in the tested run.

The root initialization effect is much smaller than the selector effect: four K=384 seeds gave KL `0.928–0.993`.

## 4. Functional exception selection dominates reconstruction-error selection

With the same K=384 root, same 2,214 exception count, same 4-bit residual format, and same total rate:

- Fisher-ranked exceptions: KL `0.947`;
- largest weight-residual-norm exceptions: KL `3.667`;
- three random matched selectors: KL `3.24–3.51`.

The residual-norm selector had **better weight NMSE** (`~0.652`) than the Fisher selector (`~0.676`) while having far worse functional quality.

The effect persisted across four different held-out story themes. Fisher-selector KL ranged approximately `0.50–0.89`; residual-norm-selector KL ranged `3.33–3.69`. This is direct evidence that token-level private exceptions should be priced by functional leverage rather than raw reconstruction error.

## 5. Scale series for the same functional root/exception scheme

With the same K=384 / 4-bit-root / 4-bit-Fisher-exception design at ~0.25 embedding bit/scalar:

| model | root bps | exceptions | exception fraction | mean KL |
|---|---:|---:|---:|---:|
| 1M | 0.1814 | 766 | 1.52% | 1.280 |
| 3M | 0.1060 | 1,703 | 3.39% | 0.733 |
| 8M | 0.0683 | 2,214 | 4.41% | 0.860 |
| 28M | 0.0494 | 2,481 | 4.94% | 0.621 |

The trend is not monotone, so this is **not** a scaling law. However the fixed root cost clearly amortizes with hidden dimension, and the largest checkpoint achieved the best tested KL under this fixed scheme. On 28M, a K sweep found KL values about `0.603 / 0.588 / 0.578 / 0.579 / 0.553 / 0.632` for K=`128/256/384/512/768/1024`, respectively; different quality witnesses do not select exactly the same K.

## 6. Task-aware weight assignment transfers beyond embedding

On the 8M checkpoint, a shared K=128 block-VQ codebook for WQ was held fixed while only the assignment metric was changed from Euclidean to a calibration-derived diagonal functional importance metric. Across all eight WQ layers, holdout mean KL fell from approximately `0.01405` to `0.01107` (about 21% reduction), with 7/8 layers improving, even though weight NMSE worsened in every layer.

The same simple metric is **not universal across roles**: on one audited layer it improved Q/V/O but worsened K. Therefore task distortion must remain role-aware rather than using one fixed Fisher-like rule for every tensor.

## 7. Joint-intervention boundary

On the tested 8M probe, embedding compression plus one WQ intervention was close to additive in KL. The measured interaction terms were about `+0.0077` (layer 3) and `+0.00005` (layer 7). Joint WQ layer-3 + layer-7 interaction was about `+0.00096` KL. Thus nonseparable task distortion can occur in general, but it should not be assumed to be large for every candidate combination.

## Boundaries

Do not infer from this note that:

- any of these checkpoints has been compressed 64x with acceptable quality;
- the manual local GPT-Neo replay is equivalent to an independently validated Transformers replay;
- the short text probes are benchmark-quality evaluation;
- the K/root/exception settings transfer to SmolLM2;
- individual KL costs are generally additive;
- model size monotonically improves compressibility.

The strongest architectural conclusion is narrower: **at sub-bit embedding rates, a shared functional root plus sparse functionally priced private exceptions is dramatically better than uniform weight reconstruction, and the fixed shared cost amortizes as representation dimension grows.**